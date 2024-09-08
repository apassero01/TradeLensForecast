import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from Transformer import  TransformerEncoder, TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, encoder_input_dim, decoder_input_dim, dropout=.1):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout=dropout, name="transformer_encoder")
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout=dropout, name="transformer_decoder")

        self.input_projection = nn.Linear(encoder_input_dim, d_model)
        self.decoder_input_projection = nn.Linear(decoder_input_dim, d_model)

        # Layer for predicting the continuation signal
        self.continuation_head = nn.Linear(d_model, 1)  # Output a single value per time step

        self.traversable_layers = [self.encoder, self.decoder]

        print(f"Transformer initialized with {len(self.traversable_layers)} layers")

    def forward(self, encoder_input, decoder_input):
        # Project inputs to model dimension
        encoder_input = self.input_projection(encoder_input)
        decoder_input = self.decoder_input_projection(decoder_input)

        # Pass through encoder and decoder
        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)

        # Predict continuation signal for each time step
        continuation_signal = torch.sigmoid(self.continuation_head(encoder_output))

        return decoder_output, continuation_signal

    def inference(self, encoder_input, start_token, max_len):
        encoder_input = self.input_projection(encoder_input)
        encoder_output = self.encoder(encoder_input)

        generated_sequence = []
        continuation_signals = []
        decoder_input = start_token.unsqueeze(0)

        for _ in range(max_len):
            decoder_output = self.decoder(decoder_input, encoder_output)

            # Predict the next token in the sequence
            next_token = decoder_output[:, -1, :]  # Last output in the sequence
            generated_sequence.append(next_token)

            # Predict the continuation signal
            continuation_signal = torch.sigmoid(self.continuation_head(next_token)).item()
            continuation_signals.append(continuation_signal)

            if continuation_signal < 0.5:  # Stop if the continuation signal is below threshold
                break

            # Update decoder input with the predicted token
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

        return torch.cat(generated_sequence, dim=0), torch.tensor(continuation_signals)


def continuous_loss(y_pred, y_true, continuation_signals, lambda_penalty=1.0):
    batch_size, max_length, _ = y_pred.size()

    total_loss = 0
    for i in range(batch_size):
        # Convert the boolean tensor to a float tensor so argmax works properly
        stop_index = torch.argmax((continuation_signals[i] <= 0.5).float()).item()

        # If stop_index is 0 and the first continuation_signal is > 0.5, it means the model predicts continuation throughout
        if stop_index == 0 and continuation_signals[i][0] > 0.5:
            stop_index = max_length  # Use the entire sequence

        # Compute the sequence loss up to the stopping point
        if stop_index == 0:
            # Apply the maximum early stop penalty because the model stopped before any prediction
            early_stop_penalty = torch.tensor(lambda_penalty * y_true.size(1))

            seq_loss = torch.tensor(0.0)  # No sequence loss since nothing was predicted
        else:

            seq_loss = F.mse_loss(y_pred[i, :stop_index, :], y_true[i, :stop_index, :], reduction='mean')
            early_stop_penalty = lambda_penalty * (y_true.size(1) - stop_index)

        total_loss += seq_loss + early_stop_penalty

    return total_loss / batch_size

