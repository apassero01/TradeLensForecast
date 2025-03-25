import math
import torch
import torch.nn as nn

from models.BaseModel import BaseLayer
from models.attention import MultiHeadAttention
from models.utils import PositionalEncoding, generate_target_mask


class EncoderToRNNWithMultiHeadAttention(BaseLayer):
    def __init__(self, d_model, dff, num_heads, output_seq_length, name = "DecoderRNN"):
        super(EncoderToRNNWithMultiHeadAttention, self).__init__()

        self.name = name

        self.num_heads = num_heads
        self.output_seq_length = output_seq_length
        # RNN layer (LSTM or GRU)
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

        # Multi-head attention mechanism (your implementation)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        self.output_sequence_length = output_seq_length

        # Linear layer to project the concatenated (rnn_hidden + attention_output) to the final output
        self.fc_out = nn.Linear(d_model*2, 100)  # Output shape will be (batch_size, output_seq_length, 2)

        self.traversable_layers = nn.ModuleList([self.multi_head_attention])

    def forward(self, encoder_output):
        if self.save_input:
            self.input = encoder_output
        # encoder_output shape: (batch_size, input_seq_length, d_model)
        batch_size, input_seq_length, d_model = encoder_output.size()

        # Initialize the hidden state and cell state for the RNN (LSTM)
        h_0 = torch.zeros(1, batch_size, d_model).to(encoder_output.device)
        c_0 = torch.zeros(1, batch_size, d_model).to(encoder_output.device)

        # Pass the encoder output through the RNN to get the hidden states
        rnn_output, (h_n, c_n) = self.rnn(encoder_output,
                                          (h_0, c_0))  # rnn_output: (batch_size, input_seq_length, dff)

        # Initialize an empty list to store the outputs
        outputs = []

        for t in range(self.output_seq_length):
            # At each step, use the last hidden state from the RNN (rnn_output[:, -1, :]) as the query
            rnn_hidden = rnn_output[:, t, :]  # (batch_size, dff)

            # Use multi-head attention with RNN hidden state as the query, and encoder output as key and value
            attention_output = self.multi_head_attention(Q=rnn_hidden.unsqueeze(1), K=encoder_output,
                                                         V=encoder_output)  # (batch_size, 1, d_model)

            attention_output = attention_output.squeeze(1)  # Remove the singleton dimension (batch_size, d_model)

            # Concatenate the RNN hidden state with the attention output
            combined_vector = torch.cat([rnn_hidden, attention_output], dim=-1)  # (batch_size, dff + d_model)

            # Pass the combined vector through the fully connected layer to get the output (mu and log_var)
            output = self.fc_out(combined_vector)  # (batch_size, 2)
            outputs.append(output.unsqueeze(1))  # (batch_size, 1, 2)

        # Concatenate all outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # (batch_size, output_seq_length, 2)

        return outputs





class TransformerDecoder(BaseLayer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=.1, name = "transformer_decoder"):
        super(TransformerDecoder, self).__init__()

        self.name = name
        self.decoder_input_projection = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout,
                         name = self.name + ":decoder_layer"+str(i)) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, 2)

        self.output_projection = nn.Linear(1, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_length=200)

        self.d_model = d_model

        self.traversable_layers = nn.ModuleList(self.layers)
        self._init_weights()


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)



    def forward(self, x, encoder_output, target_mask=None):
        x = self.positional_encoding(x)
        target_mask = generate_target_mask(x.size(1)).to(x.device)

        for layer in self.layers:
            x = layer(x, encoder_output, target_mask)

        x = self.norm(x)
        return self.fc_out(x)

    def inference(self, encoder_output, start_token, max_len):
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # Initialize the input sequence with the start token
        # y_input = start_token.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)
        y_input = start_token[:,-1:,:]
        generated_sequence = []

        for _ in range(max_len):
            # Apply positional encoding
            y_input_pe = self.positional_encoding(y_input)

            # Create target mask
            tgt_mask = generate_target_mask(y_input_pe.size(1)).to(device)

            x = y_input_pe
            for layer in self.layers:
                x = layer(x, encoder_output, tgt_mask)
            x = self.norm(x)

            # output = self.fc_out(x[:, -1, :])
            output = self.fc_out(x)

            mu = output[:,-1:, 0] # Shape: (batch_size, 1)
            log_var = output[:,-1:, 1]# Shape: (batch_size, 1)

            generated_sequence.append(torch.cat([mu, log_var], dim=1).unsqueeze(1))  # Shape: (batch_size, 1, 2)

            mu_projected = self.output_projection(mu)
            mu_projected = mu_projected.unsqueeze(1)

            y_input = torch.cat([y_input, mu_projected], dim=1)

        predictions = torch.cat(generated_sequence, dim=1)  # Shape: (batch_size, max_len, 2)
        return predictions

class DecoderLayer(BaseLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout=.1, name = "decoder_layer"):
        super(DecoderLayer, self).__init__()

        self.name = name
        self.self_attention = MultiHeadAttention(d_model, num_heads, name = name + ":self_attention")

        self.cross_attention = MultiHeadAttention(d_model, num_heads, name = name + ":cross_attention")

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.traversable_layers =  nn.ModuleList([self.self_attention, self.cross_attention])

        self._init_weights()


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, encoder_output, target_mask):
        # self_attn_output = self.self_attention(x, x, x, mask=target_mask)
        # x = self.norm1(x + self.dropout(self_attn_output))

        # cross_attn_output = self.cross_attention(x, encoder_output, encoder_output)
        # x = self.norm2(x + self.dropout(cross_attn_output))
        if self.save_input:
            self.input = encoder_output

        self_attn_output = self.self_attention(Q=self.norm1(x),
                                                  K=self.norm1(x),
                                                  V=self.norm1(x),
                                                  mask=target_mask)
        x = x + self.dropout(self_attn_output)

        # todo ensure this is correct with norm2(x) and not norm2(self_attn_output)
        cross_attn_output = self.cross_attention(Q=self.norm2(x),
                                                    K=encoder_output,
                                                    V=encoder_output,
                                                    mask=None)

        x = x + self.dropout(cross_attn_output)

        # ff_output = self.feed_forward(x)
        # output = self.norm3(x + self.dropout(ff_output))

        # Feed-Forward Network
        ff_output = self.feed_forward(self.norm3(x))
        output = x + self.dropout(ff_output)  # Residual connection

        return output

    def get_attention(self):
        return self.cross_attention_output

    def get_output(self):
        return self.orig_output