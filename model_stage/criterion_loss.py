import numpy as np
import torch
import math
import torch
import torch.nn as nn

def criterion_medAE_loss(y_true, y_pred, penalty_weight=1.0, exp=1):
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Apply exponentiation if needed
    if exp != 1:
        absolute_errors = np.power(absolute_errors, exp)

    # Compute the median of absolute errors
    medae = np.median(absolute_errors)

    # Apply penalty weight
    loss = penalty_weight * medae

    return loss


def criterion_MAE_loss(y_true, y_pred, penalty_weight=1.0, exp=1):
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Apply exponentiation if needed
    if exp != 1:
        absolute_errors = np.power(absolute_errors, exp)

    # Compute the mean of absolute errors
    mae = np.mean(absolute_errors)

    # Apply penalty weight
    loss = penalty_weight * mae

    return loss

import numpy as np
import torch

def quantile_loss(y_true, y_pred, quantile=0.5, penalty_weight=1.0, exp=1):
    """
    Compute Quantile Loss or Huber Loss with optional penalty weighting and exponentiation.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        loss_type (str): Type of loss to compute. Options: 'quantile', 'huber' (default: 'quantile').
        quantile (float): Quantile to use for Quantile Loss (default: 0.5, the median).
        delta (float): Threshold for Huber Loss (default: 1.0).
        penalty_weight (float): Weight to scale the loss (default: 1.0).
        exp (float): Exponent to raise the errors (default: 1).

    Returns:
        float or torch.Tensor: The computed loss.
    """
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute errors
    errors = y_true - y_pred

    if quantile < 0 or quantile > 1:
        raise ValueError("Quantile must be between 0 and 1.")
    absolute_errors = np.abs(errors)
    quantile_errors = np.where(errors >= 0, quantile * absolute_errors, (1 - quantile) * absolute_errors)
    if exp != 1:
        quantile_errors = np.power(quantile_errors, exp)
    loss = np.mean(quantile_errors)

    # Apply penalty weight
    loss = penalty_weight * loss

    return loss

def huber_loss(y_true, y_pred, delta=1.0, penalty_weight=1.0, exp=1):
    """
    Compute Quantile Loss or Huber Loss with optional penalty weighting and exponentiation.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        loss_type (str): Type of loss to compute. Options: 'quantile', 'huber' (default: 'quantile').
        quantile (float): Quantile to use for Quantile Loss (default: 0.5, the median).
        delta (float): Threshold for Huber Loss (default: 1.0).
        penalty_weight (float): Weight to scale the loss (default: 1.0).
        exp (float): Exponent to raise the errors (default: 1).

    Returns:
        float or torch.Tensor: The computed loss.
    """
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute errors
    errors = y_true - y_pred

    # Huber Loss
    absolute_errors = np.abs(errors)
    quadratic_errors = 0.5 * np.square(errors)
    linear_errors = delta * (absolute_errors - 0.5 * delta)
    huber_errors = np.where(absolute_errors <= delta, quadratic_errors, linear_errors)
    if exp != 1:
        huber_errors = np.power(huber_errors, exp)
    loss = np.mean(huber_errors)


    # Apply penalty weight
    loss = penalty_weight * loss

    return loss

def criterion_MSLE_loss(y_true, y_pred, penalty_weight=1.0):
    """
    Compute Mean Squared Logarithmic Error (MSLE) with optional penalty weighting.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        penalty_weight (float): Weight to scale the loss (default: 1.0).

    Returns:
        float or torch.Tensor: The MSLE loss.
    """
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, a_min=1e-7, a_max=None)
    y_true = np.clip(y_true, a_min=1e-7, a_max=None)

    # Compute squared logarithmic errors
    squared_log_errors = np.square(np.log(y_true + 1) - np.log(y_pred + 1))

    # Compute the mean
    msle = np.mean(squared_log_errors)

    # Apply penalty weight
    loss = penalty_weight * msle

    return loss

def criterion_exponential_loss(y_true, y_pred, alpha=1.0, penalty_weight=1.0):
    """
    Compute Custom Exponential Loss with optional penalty weighting.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        alpha (float): Exponential rate (default: 1.0).
        penalty_weight (float): Weight to scale the loss (default: 1.0).

    Returns:
        float or torch.Tensor: The exponential loss.
    """
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute absolute errors
    absolute_errors = np.abs(y_true - y_pred)

    # Compute exponential loss
    exponential_loss = np.mean(np.exp(alpha * absolute_errors))

    # Apply penalty weight
    loss = penalty_weight * exponential_loss

    return loss

def criterion_cauchy_loss(y_true, y_pred, gamma=1.0, penalty_weight=1.0):
    """
    Compute Cauchy Loss with optional penalty weighting.

    Args:
        y_true (np.array or torch.Tensor): Ground truth values.
        y_pred (np.array or torch.Tensor): Predicted values.
        gamma (float): Scale parameter for Cauchy Loss (default: 1.0).
        penalty_weight (float): Weight to scale the loss (default: 1.0).

    Returns:
        float or torch.Tensor: The Cauchy loss.
    """
    # Ensure inputs are numpy arrays if they are not already
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Compute errors
    errors = y_true - y_pred

    # Compute Cauchy loss
    cauchy_loss = np.mean(np.log(1 + np.square(errors / gamma)))

    # Apply penalty weight
    loss = penalty_weight * cauchy_loss

    return loss


class MSLELoss(nn.Module):
    def __init__(self, penalty_weight=1.0, reduction='mean'):
        """
        Mean Squared Logarithmic Error (MSLE) Loss.

        Args:
            penalty_weight (float): Weight to scale the loss.
            reduction (str): 'mean', 'sum', or 'none' to specify the reduction method.
        """
        super(MSLELoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.reduction = reduction

    def forward(self, preds, target):
        """
        Args:
            preds (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: The computed MSLE loss.
        """
        # Avoid log(0) by clamping to a small positive value.
        preds = torch.clamp(preds, min=1e-7)
        target = torch.clamp(target, min=1e-7)

        # Compute squared logarithmic errors.
        # Note: Using the same order as in your original function:
        # loss = (log(target + 1) - log(preds + 1))^2
        squared_log_errors = (torch.log(target + 1) - torch.log(preds + 1)) ** 2

        # Apply reduction.
        if self.reduction == 'mean':
            loss = squared_log_errors.mean()
        elif self.reduction == 'sum':
            loss = squared_log_errors.sum()
        else:
            loss = squared_log_errors  # No reduction

        return self.penalty_weight * loss

class GuissLoss(nn.Module):
    def __init__(self, threshold=0.5, penalty_weight=1.0, reduction='mean'):
        """
        Gaussian Uncertainty-Informed Scoring Loss (GuissLoss).

        This loss computes the Gaussian negative log-likelihood assuming that the model
        outputs a predicted mean and a log variance for each forecast step.
        An additional penalty is added if the predicted variance is below a threshold,
        to discourage overconfident (too low variance) predictions.

        Args:
            threshold (float): The minimum allowable variance. If predicted variance is below this,
                               a penalty is added.
            penalty_weight (float): Weight to scale the penalty term.
            reduction (str): Specifies the reduction to apply to the output:
                             'mean', 'sum', or 'none'.
        """
        super(GuissLoss, self).__init__()
        self.threshold = threshold
        self.penalty_weight = penalty_weight
        self.reduction = reduction

    def forward(self, outputs, y_true):
        """
        Args:
            outputs (torch.Tensor): Model outputs of shape (batch_size, output_steps, 2).
                                    outputs[:, :, 0] should be the predicted means (μ).
                                    outputs[:, :, 1] should be the predicted log variances (log(σ²)).
            y_true (torch.Tensor): True target values of shape (batch_size, output_steps, 1)
                                   or (batch_size, output_steps).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Extract predicted mean (mu) and log variance (log_var)
        mu = outputs[:, :, 0]       # shape: (batch_size, output_steps)
        log_var = outputs[:, :, 1]  # shape: (batch_size, output_steps)

        # Ensure y_true is the same shape as mu
        if y_true.dim() == 3 and y_true.size(-1) == 1:
            y_true = y_true.squeeze(-1)  # shape: (batch_size, output_steps)

        # Compute variance from log variance, with a small epsilon for numerical stability
        sigma_sq = torch.exp(log_var) + 1e-6

        # Compute the Gaussian negative log-likelihood
        # Formula: 0.5 * log(2πσ²) + (y - μ)² / (2σ²)
        nll = 0.5 * torch.log(2 * math.pi * sigma_sq) + ((y_true - mu) ** 2) / (2 * sigma_sq)

        # Apply a penalty if the predicted variance is below the threshold.
        # This can help discourage the model from being overconfident.
        penalty_mask = (sigma_sq < self.threshold).float()
        penalty_term = self.penalty_weight * penalty_mask * torch.clamp(self.threshold - sigma_sq, min=0)
        nll = nll + penalty_term

        # Apply the specified reduction
        if self.reduction == 'mean':
            loss = nll.mean()
        elif self.reduction == 'sum':
            loss = nll.sum()
        else:
            loss = nll  # no reduction

        return loss