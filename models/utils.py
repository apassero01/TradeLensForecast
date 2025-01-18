import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, num_inputs, d_model):
        super(InputEmbedding, self).__init__()

        self.linear = nn.Linear(num_inputs, d_model)

    def forward(self, x):
        return self.linear(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=200):
        super(PositionalEncoding, self).__init__()

        # matrix to hold the positional encodings
        pe = torch.zeros(max_length, d_model)

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))
        print(f"PositionalEncoding initialized with shape: {self.pe.shape}")

    def forward(self, x):
        # Add positional encoding to the input tensor x  # Check the first few positional encodings
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