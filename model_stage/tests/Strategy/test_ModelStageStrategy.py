from django.test import TestCase
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_stage.strategy.ModelStageStrategy import FitModelStrategy, EvaluateModelStrategy, PredictModelStrategy, ConfigureModelStageStrategy

from typing import Any, Dict

# Assuming these strategies and related classes are defined and accessible
# from your_module import (FitModelStrategy, EvaluateModelStrategy,
#                          PredictModelStrategy, ConfigureModelStageStrategy)
# from your_module import StrategyRequestEntity, StrategyExecutor, Entity, Optimizer, CriterionEnum

# Minimal mock Entity class
class MockEntity:
    def __init__(self):
        self._attributes = {}

    def has_attribute(self, name: str) -> bool:
        return name in self._attributes

    def get_attribute(self, name: str) -> Any:
        return self._attributes[name]

    def set_attribute(self, name: str, value: Any):
        self._attributes[name] = value

# Minimal mock model
class MockModel(nn.Module):
    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# A simple mock dataloader that yields a few batches of random data
def mock_dataloader(num_batches=3, batch_size=2, input_size=10):
    for _ in range(num_batches):
        encoder_input = torch.randn(batch_size, input_size)
        y_target = torch.randn(batch_size, 1)
        yield encoder_input, y_target

# Mock StrategyRequestEntity and StrategyExecutor
class MockStrategyRequestEntity:
    def __init__(self, param_config: Dict):
        self.param_config = param_config
        self.ret_val = {}

class MockStrategyExecutor:
    pass


class TestFitModelStrategy(TestCase):
    def setUp(self):
        self.entity = MockEntity()
        self.entity.set_attribute('model', MockModel())
        self.entity.set_attribute('train_dataloader', mock_dataloader())
        self.entity.set_attribute('val_dataloader', mock_dataloader())
        self.entity.set_attribute('optimizer', optim.Adam(self.entity.get_attribute('model').parameters()))
        self.entity.set_attribute('criterion', nn.MSELoss())
        self.entity.set_attribute('device', 'cpu')

        self.strategy_request = MockStrategyRequestEntity(param_config={'epochs': 1})
        self.strategy_executor = MockStrategyExecutor()
        self.strategy = FitModelStrategy(self.strategy_executor, self.strategy_request)

    def test_fit_model_success(self):
        result = self.strategy.apply(self.entity)
        self.assertIn('status', result.ret_val)
        self.assertEqual(result.ret_val['status'], 'model_fit_completed')

    def test_fit_model_missing_param(self):
        # Remove 'epochs' to trigger error
        self.strategy_request.param_config.pop('epochs', None)
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_fit_model_missing_attribute(self):
        # Remove model to trigger error
        self.entity._attributes.pop('model')
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)


class TestEvaluateModelStrategy(TestCase):
    def setUp(self):
        self.entity = MockEntity()
        self.entity.set_attribute('model', MockModel())
        self.entity.set_attribute('val_dataloader', mock_dataloader())
        self.entity.set_attribute('criterion', nn.MSELoss())
        self.entity.set_attribute('device', 'cpu')

        self.strategy_request = MockStrategyRequestEntity(param_config={})
        self.strategy_executor = MockStrategyExecutor()
        self.strategy = EvaluateModelStrategy(self.strategy_executor, self.strategy_request)

    def test_evaluate_model_success(self):
        result = self.strategy.apply(self.entity)
        self.assertIn('val_loss', result.ret_val)

    def test_evaluate_model_missing_attribute(self):
        # Remove val_dataloader
        self.entity._attributes.pop('val_dataloader')
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)


class TestPredictModelStrategy(TestCase):
    def setUp(self):
        self.entity = MockEntity()
        self.entity.set_attribute('model', MockModel())
        # Normally, prediction_input might be a single batch of data
        prediction_input = torch.randn(2, 10)  # batch_size=2, input_size=10
        self.entity.set_attribute('eval_dataloader', prediction_input)
        self.entity.set_attribute('device', 'cpu')

        self.strategy_request = MockStrategyRequestEntity(param_config={
            'prediction_input_from_entity_name': 'eval_dataloader'
        })
        self.strategy_executor = MockStrategyExecutor()
        self.strategy = PredictModelStrategy(self.strategy_executor, self.strategy_request)

    def test_predict_model_success(self):
        result = self.strategy.apply(self.entity)
        # Predictions stored in entity as 'predictions'
        self.assertTrue(self.entity.has_attribute('predictions'))

    def test_predict_model_missing_input(self):
        # Remove input attribute
        self.entity._attributes.pop('eval_dataloader')
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_predict_model_missing_param_config_key(self):
        # Change the param to something that doesn't exist
        self.strategy_request.param_config['prediction_input_from_entity_name'] = 'non_existent_input'
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)


class TestConfigureModelStageStrategy(TestCase):
    def setUp(self):
        self.entity = MockEntity()
        self.entity.set_attribute('model', MockModel())

        # Create some mock data for testing the new data pipeline
        self.X_train_data = torch.randn(100, 10)
        self.y_train_data = torch.randn(100, 1)
        self.X_test_data = torch.randn(20, 10)
        self.y_test_data = torch.randn(20, 1)

        # Set these attributes in the entity under specific names
        self.entity.set_attribute('my_X_train', self.X_train_data)
        self.entity.set_attribute('my_y_train', self.y_train_data)
        self.entity.set_attribute('my_X_test', self.X_test_data)
        self.entity.set_attribute('my_y_test', self.y_test_data)

        # Set default param_config with optimizer, criterion, and data keys
        self.strategy_request = MockStrategyRequestEntity(param_config={
            'optimizer': 'adam',
            'criterion': 'mse',
            'X_train': 'my_X_train',
            'y_train': 'my_y_train',
            'X_test': 'my_X_test',
            'y_test': 'my_y_test'
        })
        self.strategy_executor = MockStrategyExecutor()
        self.strategy = ConfigureModelStageStrategy(self.strategy_executor, self.strategy_request)

    def test_configure_model_stage_success(self):
        result = self.strategy.apply(self.entity)
        self.assertIn('status', result.ret_val)
        self.assertEqual(result.ret_val['status'], 'model_configured')
        self.assertTrue(self.entity.has_attribute('optimizer'))
        self.assertTrue(self.entity.has_attribute('criterion'))
        self.assertIsInstance(self.entity.get_attribute('optimizer'), optim.Adam)
        self.assertIsInstance(self.entity.get_attribute('criterion'), nn.MSELoss)

        # Check dataloaders
        self.assertTrue(self.entity.has_attribute('train_dataloader'))
        self.assertTrue(self.entity.has_attribute('val_dataloader'))

        train_dl = self.entity.get_attribute('train_dataloader')
        val_dl = self.entity.get_attribute('val_dataloader')
        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)

        # Test that we can get a batch
        train_batch = next(iter(train_dl))
        self.assertEqual(len(train_batch), 2)
        self.assertEqual(train_batch[0].shape, (32, 10))  # Default batch_size=32
        self.assertEqual(train_batch[1].shape, (32, 1))

        val_batch = next(iter(val_dl))
        self.assertEqual(len(val_batch), 2)
        # Since we have 20 samples in X_test, val_dl might have only one batch less than 32
        self.assertEqual(val_batch[0].shape[1], 10)
        self.assertEqual(val_batch[1].shape[1], 1)

    def test_configure_model_stage_missing_optimizer(self):
        self.strategy_request.param_config.pop('optimizer')
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_configure_model_stage_missing_criterion(self):
        self.strategy_request.param_config.pop('criterion')
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_configure_model_stage_unsupported_optimizer(self):
        self.strategy_request.param_config['optimizer'] = 'sgd'
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_configure_model_stage_unsupported_criterion(self):
        self.strategy_request.param_config['criterion'] = 'crossentropy'
        with self.assertRaises(ValueError):
            self.strategy.apply(self.entity)

    def test_configure_model_stage_missing_X_train_key(self):
        self.strategy_request.param_config.pop('X_train')
        with self.assertRaises(ValueError) as context:
            self.strategy.apply(self.entity)
        self.assertIn("X_train specification (string key) is required", str(context.exception))

    def test_configure_model_stage_missing_entity_attribute_for_X_train(self):
        # Remove the entity attribute referenced by X_train
        self.entity._attributes.pop('my_X_train')
        with self.assertRaises(ValueError) as context:
            self.strategy.apply(self.entity)
        self.assertIn("Entity does not have attribute my_X_train for X_train", str(context.exception))

    def test_configure_model_stage_missing_X_test_key(self):
        self.strategy_request.param_config.pop('X_test')
        with self.assertRaises(ValueError) as context:
            self.strategy.apply(self.entity)
        self.assertIn("X_test specification (string key) is required", str(context.exception))

    def test_configure_model_stage_missing_entity_attribute_for_X_test(self):
        # Remove the entity attribute referenced by X_test
        self.entity._attributes.pop('my_X_test')
        with self.assertRaises(ValueError) as context:
            self.strategy.apply(self.entity)
        self.assertIn("Entity does not have attribute my_X_test for X_test", str(context.exception))