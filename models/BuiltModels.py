import math
import torch
import torch.nn as nn

from models.encoders import TransformerEncoder
from models.decoders import EncoderToRNNWithMultiHeadAttention
from models.BaseModel import BaseLayer


class Transformer(BaseLayer):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config

        self.encoder = TransformerEncoder(
            self.config['num_layers'], 
            self.config['d_model'], 
            self.config['num_heads'], 
            self.config['d_ff'], 
            dropout=self.config.get('dropout', 0.1), 
            name="transformer_encoder"
        )

        self.input_projection = nn.Linear(self.config['encoder_input_dim'], self.config['d_model'])

        self.output_mechanism = EncoderToRNNWithMultiHeadAttention(
            self.config['d_model'],
            self.config['d_ff'],
            self.config['num_heads'],
            self.config['decoder_input_dim']
        )

        self.traversable_layers = [self.encoder, self.output_mechanism]

        self._init_weights()

        print(f"Transformer initialized with {len(self.traversable_layers)} layers")

    def forward(self, encoder_input, decoder_input = None, target_mask=None):
        encoder_input = self.input_projection(encoder_input)

        encoder_output = self.encoder(encoder_input)

        output = self.output_mechanism(encoder_output)

        return output

    def _init_weights(self):
        # Apply Xavier Initialization to all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)