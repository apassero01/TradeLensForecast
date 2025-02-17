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



def gaussian_nll_loss(outputs, y_true, threshold = .5):
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


    return loss


class MinOfNSequenceLoss(nn.Module):
    """
    Custom MSE-like loss:
    - 'preds' has shape (batch, seq_length, n)
    - 'target' has shape (batch, seq_length, 1)

    We compute the squared error between each of the n predictions and the target,
    then take the min error across n for each time step, and finally reduce over
    the sequence and the batch.
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        # 'reduction' can be 'mean', 'sum', or 'none'
        self.reduction = reduction

    def forward(self, preds, target):
        """
        :param preds:  (batch, seq_length, n)
        :param target: (batch, seq_length, 1)
        :return:       scalar loss by default (if reduction='mean' or 'sum'),
                       or shape (batch, seq_length) if reduction='none'
        """

        # 1) Expand target across the n dimension for easy broadcasting
        #    so target_expanded matches preds' shape in the last dim.
        #    This becomes (batch, seq_length, n).
        target_expanded = target.expand(-1, -1, preds.size(2))

        # 2) Compute squared errors -> shape (batch, seq_length, n)
        errors = (preds - target_expanded) ** 2

        # 3) Take the min across the n dimension -> shape (batch, seq_length)
        min_errors, _ = torch.min(errors, dim=2)

        # 4) Reduce over sequence and batch depending on 'reduction'
        if self.reduction == 'mean':
            return min_errors.mean()  # average over (batch * seq_length)
        elif self.reduction == 'sum':
            return min_errors.sum()  # sum over (batch * seq_length)
        else:
            # no reduction -> return shape (batch, seq_length)
            return min_errors