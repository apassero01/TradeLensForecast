import torch
import torch.nn as nn
import math

from numpy import dtype
from tensorflow.python.layers.core import dropout
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# https://www.kaggle.com/datasets/aryankhatana/sam-optimizer-pytorch/data?select=sam.py

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, encoder_input_dim, decoder_input_dim, dropout=.1):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout=dropout, name = "transformer_encoder")
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout=dropout, name = "transformer_decoder")

        self.input_projection = nn.Linear(encoder_input_dim, d_model)
        self.decoder_input_projection = nn.Linear(decoder_input_dim, d_model)

        self.traversable_layers = [self.encoder, self.decoder]

        self._init_weights()

        print(f"Transformer initialized with {len(self.traversable_layers)} layers")

    def forward(self, encoder_input, decoder_input, target_mask=None):
        encoder_input = self.input_projection(encoder_input)
        decoder_input = self.decoder_input_projection(decoder_input)

        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output)

        return decoder_output

    def inference(self, encoder_input, start_token, max_len):
        encoder_input = self.input_projection(encoder_input)

        encoder_output = self.encoder(encoder_input)
        return self.decoder.inference(encoder_output, start_token, max_len)

    def _init_weights(self):
        # Apply Xavier Initialization to all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_length=200, dropout=.1, name = "transformer_encoder"):
        super(TransformerEncoder, self).__init__()

        self.name = name

        print(f"Initializing TransformerEncoder with {num_layers} layers")
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout,
                         name = self.name +":encoder_layer"+str(i)) for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        print(f"TransformerEncoder initialized with {num_layers} layers")

        self.traversable_layers = nn.ModuleList(self.encoder_layers)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=.1, name = "encoder_layer"):
        super(EncoderLayer, self).__init__()

        self.name = name
        self.multi_head_attention = ChannelWiseMultiHeadAttention(d_model, num_heads, name = name + ":multi_head_attention")

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Tanh(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.traversable_layers = nn.ModuleList([self.multi_head_attention])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, x):
        ## pre norm
        # self.attn_output = self.multi_head_attention(x,x,x)
        # x = self.norm1(x + self.dropout(self.attn_output))
        #
        # ff_output = self.feed_forward(x)
        #
        # output = self.norm2(x + self.dropout(ff_output))

        ## post norm
        # print(f"EncoderLayer {self.name} input shape: {x.shape}")
        x_norm = self.norm1(x)  # Pre-Norm before attention
        # print(f"EncoderLayer {self.name} norm1 shape: {x_norm.shape}")
        self.attn_output = self.multi_head_attention(x_norm, x_norm, x_norm)
        # print(f"EncoderLayer {self.name} attn_output shape: {self.attn_output.shape}")
        x = x + self.dropout(self.attn_output)  # Residual connection

        # Feed-Forward with Pre-Norm
        x_norm = self.norm2(x)  # Pre-Norm before feed-forward
        ff_output = self.feed_forward(x_norm)
        output = x + self.dropout(ff_output)

        return output

    def get_attention(self):
        return self.attn_output




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, name = "multi_head_attention"):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dk = d_model // num_heads
        self.name = name

        if self.dk * num_heads != d_model:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.traversable_layers = []
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        num_features = Q.size(2)

        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        Q = Q.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)

        # Q = self.query(Q).permute(0, 2, 1)  # shape: (batch_size, num_features, seq_len)
        # K = self.key(K).permute(0, 2, 1)
        # V = self.value(V).permute(0, 2, 1)
        #
        # Q = Q.view(batch_size, num_features, self.num_heads, self.dk).transpose(1, 2)
        # K = K.view(batch_size, num_features, self.num_heads, self.dk).transpose(1, 2)
        # V = V.view(batch_size, num_features, self.num_heads, self.dk).transpose(1, 2)


        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32))
        if mask is not None:

            attention_scores = attention_scores.masked_fill(mask == True, float('-inf'))


        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        self.output = self.out(attention_output)


        return self.output

    def get_attention(self):
        return self.output

class InputEmbedding(nn.Module):
    def __init__(self, num_inputs, d_model):
        super(InputEmbedding, self).__init__()

        self.linear = nn.Linear(num_inputs, d_model)

    def forward(self, x):
        return self.linear(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length = 200):
        super(PositionalEncoding, self).__init__()

        # matrix to hold the positional encodings
        pe = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)
        print(f"PositionalEncoding initialized with shape: {self.pe.shape}")

    def forward(self, x):
        # Add positional encoding to the input tensor x
        return x + self.pe[:, :x.size(1), :]


def generate_target_mask(size):
    """
    Generates a target mask (look-ahead mask) to prevent attending to future tokens.

    Args:
    - size: The size of the target sequence (length of `y` values).

    Returns:
    - A target mask tensor with shape (size, size).
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=.1, name = "transformer_decoder"):
        super(TransformerDecoder, self).__init__()

        self.name = name
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout,
                         name = self.name + ":decoder_layer"+str(i)) for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, 1)

        self.positional_encoding = PositionalEncoding(d_model, max_length=200)

        self.d_model = d_model

        self.traversable_layers = nn.ModuleList(self.layers)
        self._init_weights()


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)



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
        y_input = start_token.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, 1, d_model)
        generated_sequence = []

        for _ in range(max_len):
            # Apply positional encoding
            y_input = self.positional_encoding(y_input)

            # Create target mask
            tgt_mask = generate_target_mask(y_input.size(1)).to(device)

            # Pass through decoder layers
            for layer in self.layers:
                y_input = layer(y_input, encoder_output, tgt_mask)

            y_input = self.norm(y_input)

            # Predict the next value
            next_value = self.fc_out(y_input[:, -1, :])  # Shape: (batch_size, 1)

            # Collect the predicted value
            generated_sequence.append(next_value.unsqueeze(1))  # Shape: (batch_size, 1, 1)

            # Prepare the next input
            next_value_expanded = next_value.unsqueeze(1)  # Shape: (batch_size, 1, 1)
            next_value_expanded = next_value_expanded.expand(batch_size, 1,
                                                             self.d_model)  # Expand to (batch_size, 1, d_model)
            y_input = torch.cat([y_input, next_value_expanded], dim=1)  # Append the new token

        # Concatenate along the sequence dimension and return as a tensor
        return torch.cat(generated_sequence, dim=1)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=.1, name = "decoder_layer"):
        super(DecoderLayer, self).__init__()

        self.name = name
        self.self_attention = MultiHeadAttention(d_model, num_heads, name = name + ":self_attention")

        self.cross_attention = MultiHeadAttention(d_model, num_heads, name = name + ":cross_attention")

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Tanh(),
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
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, x, encoder_output, target_mask):
        # self_attn_output = self.self_attention(x, x, x, mask=target_mask)
        # x = self.norm1(x + self.dropout(self_attn_output))


        # cross_attn_output = self.cross_attention(x, encoder_output, encoder_output)
        # x = self.norm2(x + self.dropout(cross_attn_output))

        self_attn_output = self.self_attention(Q=self.norm1(x),
                                                  K=self.norm1(x),
                                                  V=self.norm1(x),
                                                  mask=target_mask)
        x = x + self.dropout(self_attn_output)

        cross_attn_output = self.cross_attention(Q=self.norm2(self_attn_output),
                                                    K=encoder_output,
                                                    V=encoder_output,
                                                    mask=None)

        x = x + self.dropout(cross_attn_output)

        # ff_output = self.feed_forward(x)
        # output = self.norm3(x + self.dropout(ff_output))

        # Feed-Forward Network
        ff_output = self.feed_forward(self.norm3(x))
        output = x + self.dropout(ff_output)  # Residual connection

        self.cross_attention_output = cross_attn_output
        return output

    def get_attention(self):
        return self.cross_attention_output

class ChannelWiseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(ChannelWiseMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dk = d_model // num_heads
        self.name = name

        if self.dk * num_heads != d_model:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        # Define linear layers for query, key, value, and output
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.traversable_layers = []
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        num_features = Q.size(2)

        # Apply linear transformations to Q, K, V without permuting yet
        Q = self.query(Q)  # (batch_size, seq_len, d_model)
        K = self.key(K)    # (batch_size, seq_len, d_model)
        V = self.value(V)  # (batch_size, seq_len, d_model)

        # Permute to move features to the second dimension for channel-wise attention
        Q = Q.permute(0, 2, 1)  # (batch_size, num_features (d_model), seq_len)
        K = K.permute(0, 2, 1)  # (batch_size, num_features (d_model), seq_len)
        V = V.permute(0, 2, 1)  # (batch_size, num_features (d_model), seq_len)

        # Reshape for multi-head attention: (batch_size, num_heads, feature_head_size, seq_len)
        Q = Q.view(batch_size, self.num_heads, self.dk, seq_len).transpose(1, 2)
        K = K.view(batch_size, self.num_heads, self.dk, seq_len).transpose(1, 2)
        V = V.view(batch_size, self.num_heads, self.dk, seq_len).transpose(1, 2)

        # Compute attention scores across features (channels)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32))

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == True, float('-inf'))

        # Compute attention weights and apply them to values
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Reshape back to (batch_size, num_features, seq_len)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, num_features, seq_len)

        # Permute back to original shape (batch_size, seq_len, num_features)
        attention_output = attention_output.permute(0, 2, 1)

        # Apply final linear transformation and return output
        self.output = self.out(attention_output)
        return self.output

    def get_attention(self):
        return self.output



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        # Initialize the base optimizer (e.g., SGD, Adam)
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Compute the norm of the gradients
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)  # Scaling factor for perturbation

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Perturbation applied to parameters
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Revert the perturbation to get back to original parameters
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()  # Do the actual sharpness-aware update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        # This method is not used in SAM and intentionally left unimplemented
        raise NotImplementedError("SAM doesn't work like the other optimizers, "
                                  "you should call `first_step` and `second_step` instead.")

    def _grad_norm(self):
        # Compute the gradient norm for scaling the perturbation
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def custom_loss_with_zero_penalty(y_true, y_pred, penalty_weight=1.0):
    # Calculate the standard MSE loss
    mse_loss = nn.functional.mse_loss(y_pred, y_true)

    # Apply a penalty to predictions close to zero
    zero_penalty = torch.mean(torch.exp(-torch.abs(y_pred)))

    # Combine the MSE loss with the zero-penalty term
    total_loss = mse_loss + penalty_weight * zero_penalty

    return total_loss








