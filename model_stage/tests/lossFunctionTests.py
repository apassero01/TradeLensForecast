from django.test import TestCase
import numpy as np
import torch

# Import your loss functions here
from model_stage.criterion_loss import (
    criterion_medAE_loss,
    criterion_MAE_loss,
    quantile_loss,
    huber_loss,
    criterion_MSLE_loss,
    criterion_exponential_loss,
    criterion_cauchy_loss,
)

class LossFunctionsTest(TestCase):
    def setUp(self):
        # Common test data
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.5])
        self.y_true_tensor = torch.tensor(self.y_true)
        self.y_pred_tensor = torch.tensor(self.y_pred)

    # Test MedAE Loss
    def test_criterion_medAE_loss(self):
        loss = criterion_medAE_loss(self.y_true, self.y_pred)
        self.assertAlmostEqual(loss, 0.2, delta=0.1)

        loss_tensor = criterion_medAE_loss(self.y_true_tensor, self.y_pred_tensor)
        self.assertAlmostEqual(loss_tensor, 0.2, delta=0.1)

        loss_weighted = criterion_medAE_loss(self.y_true, self.y_pred, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.2, delta=0.1)

        loss_exp = criterion_medAE_loss(self.y_true, self.y_pred, exp=2)
        self.assertAlmostEqual(loss_exp, 0.04, delta=0.1)

    # Test MAE Loss
    def test_criterion_MAE_loss(self):
        loss = criterion_MAE_loss(self.y_true, self.y_pred)
        self.assertAlmostEqual(loss, 0.2, delta=0.01)

        loss_tensor = criterion_MAE_loss(self.y_true_tensor, self.y_pred_tensor)
        self.assertAlmostEqual(loss_tensor, 0.2, delta=0.01)

        loss_weighted = criterion_MAE_loss(self.y_true, self.y_pred, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.4, delta=0.01)

        loss_exp = criterion_MAE_loss(self.y_true, self.y_pred, exp=2)
        self.assertAlmostEqual(loss_exp, 0.064, delta=0.01)

    # Test Quantile Loss
    def test_quantile_loss(self):
        loss = quantile_loss(self.y_true, self.y_pred, quantile=0.5)
        self.assertAlmostEqual(loss, 0.1, delta=0.01)

        loss_tensor = quantile_loss(self.y_true_tensor, self.y_pred_tensor, quantile=0.9)
        self.assertAlmostEqual(loss_tensor, 0.036, delta=0.01)

        loss_weighted = quantile_loss(self.y_true, self.y_pred, quantile=0.5, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.2, delta=0.1)

        loss_exp = quantile_loss(self.y_true, self.y_pred, quantile=0.5, exp=2)
        self.assertAlmostEqual(loss_exp, 0.04, delta=0.03)

    # Test Huber Loss
    def test_huber_loss(self):
        loss = huber_loss(self.y_true, self.y_pred, delta=1.0)
        self.assertAlmostEqual(loss, 0.045, delta=0.02)

        loss_tensor = huber_loss(self.y_true_tensor, self.y_pred_tensor, delta=1.0)
        self.assertAlmostEqual(loss_tensor, 0.045, delta=0.02)

        loss_weighted = huber_loss(self.y_true, self.y_pred, delta=1.0, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.09, delta=0.03)

        loss_exp = huber_loss(self.y_true, self.y_pred, delta=1.0, exp=2)
        self.assertAlmostEqual(loss_exp, 0.002025, delta=0.01)

    # Test MSLE Loss
    def test_criterion_MSLE_loss(self):
        # Test with numpy arrays
        loss = criterion_MSLE_loss(self.y_true, self.y_pred)
        self.assertAlmostEqual(loss, 0.0028, delta=1e-5)  # Allow a tolerance of 1e-6

        # Test with torch tensors
        loss_tensor = criterion_MSLE_loss(self.y_true_tensor, self.y_pred_tensor)
        self.assertAlmostEqual(loss_tensor, 0.0028, delta=1e-5)

        # Test with penalty weight
        loss_weighted = criterion_MSLE_loss(self.y_true, self.y_pred, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.0056, delta=1e-5)

    # Test Exponential Loss
    def test_criterion_exponential_loss(self):
        loss = criterion_exponential_loss(self.y_true, self.y_pred, alpha=1.0)
        self.assertAlmostEqual(loss, 1.23, delta=0.01)

        loss_tensor = criterion_exponential_loss(self.y_true_tensor, self.y_pred_tensor, alpha=1.0)
        self.assertAlmostEqual(loss_tensor, 1.23, delta=0.01)

        loss_weighted = criterion_exponential_loss(self.y_true, self.y_pred, alpha=1.0, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 2.46, delta=0.02)

    # Test Cauchy Loss
    def test_criterion_cauchy_loss(self):
        loss = criterion_cauchy_loss(self.y_true, self.y_pred, gamma=1.0)
        self.assertAlmostEqual(loss, 0.06, delta=0.01)

        loss_tensor = criterion_cauchy_loss(self.y_true_tensor, self.y_pred_tensor, gamma=1.0)
        self.assertAlmostEqual(loss_tensor, 0.06, delta=0.01)

        loss_weighted = criterion_cauchy_loss(self.y_true, self.y_pred, gamma=1.0, penalty_weight=2.0)
        self.assertAlmostEqual(loss_weighted, 0.12, delta=0.01)