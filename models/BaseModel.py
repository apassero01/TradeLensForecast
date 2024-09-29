import torch
import torch.nn as nn
import numpy as np

class BaseLayer(nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self.input = None
        self.output = None
        self.traversable_layers = []


    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable parameters: ", params)
        print(self)
        return params

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_traversible_layers(self):
        return self.traversable_layers

