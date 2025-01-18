from copy import deepcopy

import torch
import torch.nn as nn

from models.BaseModel import BaseLayer


class MultiHeadAttention(BaseLayer):
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
        self.layer_norm = nn.LayerNorm(d_model)

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

    def forward(self, Q, K, V, mask=None):
        if self.save_input:
            self.input = deepcopy(Q)
        batch_size = Q.size(0)

        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        Q = Q.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.dk).transpose(1, 2)


        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32))
        if mask is not None:

            attention_scores = attention_scores.masked_fill(mask == True, float('-inf'))


        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(attention_output)

        output = self.layer_norm(output)

        return output


class ChannelWiseMultiHeadAttention(BaseLayer):
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

        self.layer_norm = nn.LayerNorm(d_model)

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

    def forward(self, Q, K, V, mask=None):
        if self.save_input:
            self.input = deepcopy(Q)

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
        output = self.out(attention_output)
        output = self.layer_norm(output)

        return output
