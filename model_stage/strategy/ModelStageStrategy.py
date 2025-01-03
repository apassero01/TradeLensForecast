import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.Entity import Entity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from model_stage.entities.ModelStageEntity import ModelStageEntity
from shared_utils.entities.EnityEnum import EntityEnum
from models.BuiltModels import Transformer
from model_stage.Enums.ConfigurationEnum import CriterionEnum, OptimizerEnum

class ModelStageStrategy(Strategy):
    entity_type = EntityEnum.MODEL_STAGE
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    def apply(self, entity: ModelStageEntity):
        raise NotImplementedError("Child classes must implement the 'apply' method.")

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}
    

class CreateModelStrategy(ModelStageStrategy):
    

    def apply(self, entity: ModelStageEntity):
        param_config = self.strategy_request.param_config
        model = Transformer(param_config)
        
        entity.set_attribute("model", model)

        return self.strategy_request

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if 'num_layers' not in strategy_request.param_config:
            raise ValueError("num_layers is required")
        if 'd_model' not in strategy_request.param_config:
            raise ValueError("d_model is required")
        if 'num_heads' not in strategy_request.param_config:
            raise ValueError("num_heads is required")
        if 'd_ff' not in strategy_request.param_config:
            raise ValueError("d_ff is required")
        if 'encoder_input_dim' not in strategy_request.param_config:
            raise ValueError("encoder_input_dim is required")
        if 'decoder_input_dim' not in strategy_request.param_config:
            raise ValueError("decoder_input_dim is required")
    
    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CreateModelStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'num_layers': 2,
                'd_model': 128,
                'num_heads': 8,
                'd_ff': 256,
                'encoder_input_dim': 6,
                'decoder_input_dim': 6
            }
        }
class FitModelStrategy(Strategy):
    """
    Strategy that trains (fits) a model for a given number of epochs,
    using provided training and validation dataloaders and an optional gradient clip value.
    """
    strategy_description = 'Trains the model over a specified number of epochs using given dataloaders.'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        config = strategy_request.param_config
        required_params = ['epochs']

        if not entity.has_attribute('model'):
            raise ValueError('Model not found in entity.')
        if not entity.has_attribute('train_dataloader'):
            raise ValueError('Train dataloader not found in entity.')
        if not entity.has_attribute('val_dataloader'):
            raise ValueError('Validation dataloader not found in entity.')
        if not entity.has_attribute('optimizer'):
            raise ValueError('Optimizer not found in entity.')
        if not entity.has_attribute('criterion'):
            raise ValueError('Criterion not found in entity.')
        if not entity.has_attribute('device'):
            raise ValueError('Device not found in entity (e.g., "cpu" or "cuda").')

        missing = [p for p in required_params if p not in config]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config

        model = entity.get_attribute('model')
        train_dataloader = entity.get_attribute('train_dataloader')
        val_dataloader = entity.get_attribute('val_dataloader')
        optimizer = entity.get_attribute('optimizer')
        criterion = entity.get_attribute('criterion')
        device = entity.get_attribute('device')

        epochs = config.get('epochs')
        clip_value = config.get('clip_value', None)

        model.to(device)
        print(next(model.parameters()).device)

        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_dataloader, criterion, optimizer, device, clip_value)
            val_loss = self._evaluate(model, val_dataloader, criterion, device)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        self.strategy_request.ret_val['status'] = 'model_fit_completed'
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'epochs': None,
            'clip_value': None
        }

    def _train_epoch(self, model, dataloader, criterion, optimizer, device, clip_value=None):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            encoder_input, y_target = batch
            encoder_input = encoder_input.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()
            predictions = model(encoder_input)
            loss = criterion(predictions, y_target)
            loss.backward()

            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _evaluate(self, model, dataloader, criterion, device):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                encoder_input, y_target = batch
                encoder_input = encoder_input.to(device)
                y_target = y_target.to(device)

                predictions = model(encoder_input)
                loss = criterion(predictions, y_target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


class EvaluateModelStrategy(Strategy):
    """
    Strategy that evaluates the model on a given dataloader and returns the validation loss.
    """
    strategy_description = 'Evaluates the model on a provided validation dataloader and returns the validation loss.'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('model'):
            raise ValueError('Model not found in entity.')
        if not entity.has_attribute('val_dataloader'):
            raise ValueError('Validation dataloader not found in entity.')
        if not entity.has_attribute('criterion'):
            raise ValueError('Criterion not found in entity.')
        if not entity.has_attribute('device'):
            raise ValueError('Device not found in entity.')

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        self.verify_executable(entity, self.strategy_request)

        model = entity.get_attribute('model')
        val_dataloader = entity.get_attribute('val_dataloader')
        criterion = entity.get_attribute('criterion')
        device = entity.get_attribute('device')

        model.to(device)
        val_loss = self._evaluate(model, val_dataloader, criterion, device)
        self.strategy_request.ret_val['val_loss'] = val_loss
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {}

    def _evaluate(self, model, dataloader, criterion, device):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                encoder_input, y_target = batch
                encoder_input = encoder_input.to(device)
                y_target = y_target.to(device)

                predictions = model(encoder_input)
                loss = criterion(predictions, y_target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


class PredictModelStrategy(Strategy):
    """
    Strategy that uses the model to generate predictions given an input.
    """
    strategy_description = 'Generates predictions using the model given an input tensor.'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('model'):
            raise ValueError('Model not found in entity.')
        if not entity.has_attribute(strategy_request.param_config['prediction_input_from_entity_name']):
            raise ValueError('Prediction input not found in entity.')
        if not entity.has_attribute('device'):
            raise ValueError('Device not found in entity.')

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        self.verify_executable(entity, self.strategy_request)

        model = entity.get_attribute('model')
        prediction_input = entity.get_attribute(self.strategy_request.param_config['prediction_input_from_entity_name'])
        device = entity.get_attribute('device')

        model.to(device)

        predictions = self._predict(model, prediction_input, device)
        entity.set_attribute("predictions", predictions)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            "prediction_input_from_entity_name": 'eval_dataloader'
        }

    def _predict(self, model, prediction_input, device):
        model.eval()
        X = torch.cat([X for X, y in prediction_input], dim=0)
        with torch.no_grad():
            predictions = model(X.to(device))
        # Convert predictions to CPU and numpy if necessary:
        return predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions


class ConfigureModelStageStrategy(Strategy):
    """
    Strategy to configure model components such as optimizer and criterion based on provided strings.
    The strategy uses enums to interpret the given strings and creates the respective PyTorch optimizer and criterion.
    Additionally, it uses provided attribute keys to retrieve X_train, y_train, X_test, and y_test from the entity.
    From these tensors, it constructs training and validation dataloaders and sets them on the entity.
    """
    strategy_description = 'Configures the model stage by setting optimizer, criterion, and creating train/val dataloaders.'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        config = strategy_request.param_config
        # Check optimizer and criterion
        if 'optimizer' not in config:
            raise ValueError("Optimizer specification is required.")
        if 'criterion' not in config:
            raise ValueError("Criterion specification is required.")

        # Check model
        if not entity.has_attribute('model'):
            raise ValueError("Entity must have a model before configuration.")

        # Check data keys
        required_data_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'device']
        for key in required_data_keys:
            if key not in config:
                raise ValueError(f"{key} specification (string key) is required.")

        # Check that entity has those attributes
        if not entity.has_attribute(config['X_train']):
            raise ValueError(f"Entity does not have attribute: {config['X_train']}")
        if not entity.has_attribute(config['y_train']):
            raise ValueError(f"Entity does not have attribute: {config['y_train']}")
        if not entity.has_attribute(config['X_test']):
            raise ValueError(f"Entity does not have attribute: {config['X_test']}")
        if not entity.has_attribute(config['y_test']):
            raise ValueError(f"Entity does not have attribute: {config['y_test']}")

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config

        optimizer_str = config.get('optimizer')
        criterion_str = config.get('criterion')

        # Retrieve model
        model = entity.get_attribute('model')
        optimizer = self._create_optimizer(optimizer_str, model)
        criterion = self._create_criterion(criterion_str)

        entity.set_attribute("optimizer", optimizer)
        entity.set_attribute("criterion", criterion)
        entity.set_attribute("criterion_name", criterion_str)
        entity.set_attribute("optimizer_name", optimizer_str)

        # Retrieve data using the keys
        X_train = entity.get_attribute(config['X_train'])
        y_train = entity.get_attribute(config['y_train'])
        X_test = entity.get_attribute(config['X_test'])
        y_test = entity.get_attribute(config['y_test'])

        # Construct DataLoader objects
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        if isinstance(y_test, np.ndarray):
            y_test = torch.from_numpy(y_test).float()

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        entity.set_attribute('train_dataloader', train_dataloader)
        entity.set_attribute('val_dataloader', val_dataloader)

        entity.set_attribute('device', config['device'])

        self.strategy_request.ret_val['status'] = 'model_configured'
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'optimizer': "adam",
            'criterion': "mse",
            'X_train': "X_train_scaled",
            'y_train': 'y_train_scaled',
            'X_test': 'X_test_scaled',
            'y_test': 'y_test_scaled',
            'device': 'mps'
        }

    def _create_optimizer(self, optimizer_str: str, model):
        optimizer_str = optimizer_str.lower()
        if optimizer_str == OptimizerEnum.OPTIMIZER_ADAM.value:
            return optim.Adam(model.parameters(), lr=0.001)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_str}")

    def _create_criterion(self, criterion_str: str):
        criterion_str = criterion_str.lower()
        if criterion_str == CriterionEnum.CRITERION_MSE.value:
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_str}")