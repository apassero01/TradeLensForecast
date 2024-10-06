from django.test import TestCase
import numpy as np

from training_manager.EvaluationService import EvaluationService
from training_manager.models import TrainingSession, Trainer


class EvaluationServiceTestCase(TestCase):

    def setUp(self):
        self.training_session = TrainingSession.objects.create(X_features=['open', 'close'], y_features=['close'], sequence_params={'window_size': 10})
        self.trainer = Trainer.objects.create(model_params={'hidden_size': 64}, model_weights_dir='model_weights', training_session=self.training_session)
        self.eval_service = EvaluationService()

    def test_directional_accuracy(self):

        # Test case 1: Simple example where all directions match
        y_pred = np.array([[0.5, -0.2, 1.1], [-0.5, -0.8, 0.9]])  # Shape: (2, 3)
        y_true = np.array([[0.3, -0.1, 1.0], [-0.4, -0.9, 1.0]])  # Shape: (2, 3)
        expected_accuracy = np.array([1.0, 1.0, 1.0])
        accuracy = self.eval_service.compute_dir_accruacy(y_pred, y_true)
        np.testing.assert_almost_equal(accuracy, expected_accuracy)

        # Test case 2: Half the directions match in the second example
        y_pred = np.array([[0.5, -0.2, 1.1], [-0.5, 0.8, -0.9]])  # Shape: (2, 3)
        y_true = np.array([[0.3, -0.1, 1.0], [-0.4, -0.9, 1.0]])  # Shape: (2, 3)
        expected_accuracy = np.array([1.0, 0.5, 0.5])  # t=0: 100%, t=1: 50%, t=2: 50%
        accuracy = self.eval_service.compute_dir_accruacy(y_pred, y_true)
        np.testing.assert_almost_equal(accuracy, expected_accuracy)

        # Test case 3: No matching directions
        y_pred = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])  # Shape: (2, 3)
        y_true = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])  # Shape: (2, 3)
        expected_accuracy = np.array([0.0, 0.0, 0.0])
        accuracy = self.eval_service.compute_dir_accruacy(y_pred, y_true)
        np.testing.assert_almost_equal(accuracy, expected_accuracy)