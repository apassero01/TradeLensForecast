import math
import torch
import torch.nn as nn

from models.BaseModel import BaseLayer
from models.attention import MultiHeadAttention, ChannelWiseMultiHeadAttention
from models.utils import PositionalEncoding


class TransformerEncoder(BaseLayer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_length=200, dropout=.1, name = "transformer_encoder"):
        super(TransformerEncoder, self).__init__()

        self.name = name

        print(f"Initializing TransformerEncoder with {num_layers} layers")
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        # self.encoder_layers = nn.ModuleList([
        #     EncoderLayer(d_model, num_heads, d_ff, dropout,
        #                  name = self.name +":encoder_layer"+str(i)) for i in range(num_layers)
        # ])
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                attention_type = "temporal"
            else:
                attention_type = "channel"
            self.encoder_layers.append(EncoderLayer(d_model, num_heads, d_ff, dropout,
                                                     name = self.name +":encoder_layer"+str(i),
                                                     attention_type = attention_type))


        self.dropout = nn.Dropout(dropout)
        print(f"TransformerEncoder initialized with {num_layers} layers")

        self.traversable_layers = nn.ModuleList(self.encoder_layers)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        return x

class EncoderLayer(BaseLayer):
    def __init__(self, d_model, num_heads, d_ff, dropout=.1, name = "encoder_layer", attention_type = "temporal"):
        super(EncoderLayer, self).__init__()

        self.name = name
        # self.multi_head_attention = MultiHeadAttention(d_model, num_heads, name = name + ":multi_head_attention")
        if attention_type == "temporal":
            self.multi_head_attention = MultiHeadAttention(d_model, num_heads, name = name + ":multi_head_attention")
        else:
            self.multi_head_attention = ChannelWiseMultiHeadAttention(d_model, num_heads, name = name + ":channel_multi_head_attention")

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(),
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
                # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        self.input = x.clone()

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
        self.output = output

        return output

