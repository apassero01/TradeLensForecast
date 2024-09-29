import torch
import torch.nn as nn
from django.utils.timezone import override
from tensorflow.python.ops.gen_array_ops import deep_copy

from collections import deque

from models.ModelScraper import ModelScraper


class Trainer:
    def __init__(self, model, optimizer, criterion, device, max_length, start_token_value=0.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_token_value = start_token_value
        self.max_length = max_length
        self.model_scraper = ModelScraper(model)

    def to_device(self):
        """Move the models to the specified device."""
        self.model.to(self.device)

    def add_start_token(self, y_target):
        """Add start token to the beginning of the target sequence."""
        batch_size = y_target.size(0)
        seq_len = y_target.size(1)
        start_token = torch.zeros((batch_size, 1, 1)).to(y_target.device)
        # makethe token -999
        # start_token.fill_(-999)

        # Ensure y_target has three dimensions before slicing
        if y_target.dim() == 2:  # If y_target is (batch_size, seq_len)
            y_target = y_target.unsqueeze(-1)  # Make it (batch_size, seq_len, 1)

        # Concatenate the start token with y_target along the sequence dimension
        decoder_input = torch.cat([start_token, y_target[:, :-1, :]], dim=1)
        return decoder_input

    def train_epoch(self, dataloader, clip_value=None):
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            encoder_input, y_target = [x.to(self.device) for x in batch]
            decoder_input = self.add_start_token(y_target)

            self.optimizer.zero_grad()

            # Get the sequence predictions and continuation signals from the models
            self.sequence_predictions = self.model(encoder_input, decoder_input)

            # loss = self.criterion(torch.cumsum(sequence_predictions.squeeze(-1), dim=1), torch.cumsum(y_target.squeeze(-1), dim=1))

            # Calculate the custom loss
            loss = self.criterion(self.sequence_predictions, y_target)

            loss.backward()

            if clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def train_epoch_contin_signal(self, dataloader, clip_value=None):
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            encoder_input, y_target = [x.to(self.device) for x in batch]
            decoder_input = self.add_start_token(y_target)

            self.optimizer.zero_grad()

            # Get the sequence predictions and continuation signals from the models
            sequence_predictions, continuation_signals = self.model(encoder_input, decoder_input)

            # Calculate the custom loss
            loss = self.criterion(sequence_predictions, y_target, continuation_signals)

            loss.backward()

            if clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def train_epoch_SAM(self, dataloader):
        self.model.train()  # Set the models to training mode
        epoch_loss = 0  # Initialize the epoch loss

        for batch in dataloader:
            # Move data to the appropriate device
            encoder_input, y_target = [x.to(self.device) for x in batch]
            decoder_input = self.add_start_token(y_target)  # Prepare the decoder input

            # First forward-backward pass
            self.optimizer.zero_grad()  # Clear gradients from the previous iteration
            output = self.model(encoder_input, decoder_input)  # Forward pass through the models
            loss = self.criterion(output, y_target)  # Compute the loss
            loss.backward()  # Compute gradients for the first step
            self.optimizer.first_step(zero_grad=True)  # SAM's first step with gradient ascent

            # Second forward-backward pass
            output = self.model(encoder_input, decoder_input)  # Forward pass with perturbed parameters
            loss = self.criterion(output, y_target)  # Compute the loss again
            loss.backward()  # Compute gradients for the second step
            self.optimizer.second_step(zero_grad=True)  # SAM's second step with gradient descent

            epoch_loss += loss.item()  # Accumulate the loss for the current batch

        # Calculate the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)

        return avg_epoch_loss  # Return the average epoch loss


    def evaluate(self, dataloader):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                encoder_input, y_target = [x.to(self.device) for x in batch]

                output = self.model(encoder_input, None)
                loss = self.criterion(output, y_target)
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def predict(self, encoder_input):
        self.model.eval()  # Set the models to evaluation mode

        with torch.no_grad():
            # encoder_input = encoder_input.to(self.device)

            # Prepare the start token with the correct shape

            # Perform inference to get the predicted output sequence
            predictions = self.model(encoder_input, None)

        print(f"Predictions shape: {predictions.shape}")
        return predictions


    def fit(self, train_dataloader, val_dataloader, epochs, clip_value=None):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader, clip_value)
            val_loss = self.evaluate(val_dataloader)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

    def fit_SAM(self, train_dataloader, val_dataloader, epochs, clip_value=None):
        for epoch in range(epochs):
            train_loss = self.train_epoch_SAM(train_dataloader)
            val_loss = self.evaluate(val_dataloader)

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")


    def log_gradients_with_names(self, inputs, targets, criterion=nn.MSELoss()):
        # Set the models to training mode
        self.model.train()

        # Zero the gradients before running the backward pass
        self.model.zero_grad()

        decoder_input = self.add_start_token(targets)

        # Perform a forward pass
        outputs = self.model(inputs, decoder_input)

        # Calculate the loss
        loss = self.criterion(outputs, targets)

        # Perform a backward pass to compute the gradients
        loss.backward()

        # Get gradients with names
        gradients_with_names = self.get_gradients_with_names()

        return gradients_with_names

    def generate_model_architecture(self):
        """Print the models architecture."""

        layer_names = []
        next_layers = self.model.traversable_layers

        # make list a queue to traverse the layers
        queue = deque(next_layers)

        while len(queue) > 0:
            layer = queue.popleft()
            layer_names.append(layer.name)
            queue.extend(layer.traversable_layers)
            # next_layers.extend(layer.traversable_layers)
            # next_layers.remove(layer)

        output = "Model Architecture:\n"
        output += "------------------\n"
        for name in layer_names:
            output += f"'{name}'\n"

        print(output)



    def get_gradients_with_names(self):
        gradients_with_names = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients_with_names.append((name, param.grad))
        return gradients_with_names


    def scrape_model(self, retrieval_dict):
        self.model_scraper.scrape_model(retrieval_dict)
        return self.model_scraper.output_dict
