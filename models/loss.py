import math
import torch
import torch.nn as nn

def custom_loss_with_zero_penalty(y_true, y_pred, penalty_weight=1.0):
    # Calculate the standard MSE loss
    mse_loss = nn.functional.mse_loss(y_pred, y_true)

    # Apply a penalty to predictions close to zero
    zero_penalty = torch.mean(torch.exp(-torch.abs(y_pred)))

    # Combine the MSE loss with the zero-penalty term
    total_loss = mse_loss + penalty_weight * zero_penalty

    return total_loss



def gaussian_nll_loss(outputs, y_true, threshold = .5, penalty_factor = 1000):
    """
    Computes the Gaussian negative log-likelihood loss.

    Args:
        outputs (torch.Tensor): Model outputs of shape (batch_size, output_steps, 2).
                                outputs[:, :, 0] contains the predicted means (mu).
                                outputs[:, :, 1] contains the predicted log variances (log_var).
        y_true (torch.Tensor): True target values of shape (batch_size, output_steps, 1).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Step 1: Extract mu and log_var
    mu = outputs[:, :, 0]        # Shape: (batch_size, output_steps)
    log_var = outputs[:, :, 1]   # Shape: (batch_size, output_steps)

    # Step 2: Ensure y_true matches the shape of mu
    y_true = y_true.squeeze(-1)  # Shape: (batch_size, output_steps)

    # Step 3: Compute variance (sigma^2) from log variance
    sigma_sq = torch.exp(log_var)  # Shape: (batch_size, output_steps)

    # Optional: Add a small epsilon for numerical stability
    epsilon = 1e-6
    sigma_sq = sigma_sq + epsilon

    # Step 4: Compute the negative log-likelihood
    nll = 0.5 * torch.log(2 * math.pi * sigma_sq) + ((y_true - mu) ** 2) / (2 * sigma_sq)

    # Step 5: Compute the mean loss over all elements
    loss = torch.mean(nll)

    # if mu is close to zero, add a penalty term
    penalty = torch.where(mu.abs() < threshold, penalty_factor, torch.zeros_like(mu))

    return loss + penalty.mean()