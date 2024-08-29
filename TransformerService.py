import torch
import torch.nn as nn
from tensorflow.python.ops.gen_array_ops import deep_copy

from collections import deque

class Trainer:
    def __init__(self, model, optimizer, criterion, device, max_length, start_token_value=0.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_token_value = start_token_value
        self.max_length = max_length

    def to_device(self):
        """Move the model to the specified device."""
        self.model.to(self.device)

    def add_start_token(self, y_target):
        """Add start token to the beginning of the target sequence."""
        batch_size = y_target.size(0)
        seq_len = y_target.size(1)
        start_token = torch.zeros((batch_size, 1, 1)).to(y_target.device)

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

            output = self.model(encoder_input, decoder_input)

            loss = self.criterion(output, y_target)

            loss.backward()
            if clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def train_epoch_SAM(self, dataloader):
        self.model.train()  # Set the model to training mode
        epoch_loss = 0  # Initialize the epoch loss

        for batch in dataloader:
            # Move data to the appropriate device
            encoder_input, y_target = [x.to(self.device) for x in batch]
            decoder_input = self.add_start_token(y_target)  # Prepare the decoder input

            # First forward-backward pass
            self.optimizer.zero_grad()  # Clear gradients from the previous iteration
            output = self.model(encoder_input, decoder_input)  # Forward pass through the model
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

                start_token = torch.zeros(self.model.decoder.d_model).to(self.device)

                output = self.model.inference(encoder_input, start_token, self.max_length)
                loss = self.criterion(output, y_target)
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def predict(self, encoder_input):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            # encoder_input = encoder_input.to(self.device)

            # Prepare the start token with the correct shape
            start_token = torch.zeros(self.model.decoder.d_model).to(self.device)

            # Perform inference to get the predicted output sequence
            predictions = self.model.inference(encoder_input, start_token, self.max_length)
            last_attn = self.model.decoder.layers[-1].cross_attention_output

        print(f"Predictions shape: {predictions.shape}")
        print(f"Last attention shape: {last_attn.shape}")
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
        # Set the model to training mode
        self.model.train()

        # Zero the gradients before running the backward pass
        self.model.zero_grad()

        decoder_input = self.add_start_token(targets)

        # Perform a forward pass
        outputs = self.model(inputs, decoder_input)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Perform a backward pass to compute the gradients
        loss.backward()

        # Get gradients with names
        gradients_with_names = self.get_gradients_with_names()

        return gradients_with_names

    def generate_model_architecture(self):
        """Print the model architecture."""

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

    def retreive_attention_scores(self, retrieval_list):

        output_dict = {}
        next_layers = self.model.traversable_layers
        queue = deque(next_layers)

        while len(queue) > 0:
            layer = queue.popleft()

            if layer.name in retrieval_list:
                output_dict[layer.name] = layer.get_attention()
            queue.extend(layer.traversable_layers)

        return output_dict


    def get_gradients_with_names(self):
        gradients_with_names = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients_with_names.append((name, param.grad))
        return gradients_with_names
