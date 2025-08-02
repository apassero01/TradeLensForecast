import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from model_stage.criterion_loss import MSLELoss, GuissLoss, MinOfNSequenceLoss, SoftInverseProfitLoss, SequenceNLLLoss
from model_stage.strategy.RL.RLUtils import TradingEnv, ReplayBuffer, epsilon_by_frame
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.Entity import Entity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from model_stage.entities.ModelStageEntity import ModelStageEntity
from shared_utils.entities.EnityEnum import EntityEnum
from models.BuiltModels import Transformer, DQN
from model_stage.Enums.ConfigurationEnum import CriterionEnum, OptimizerEnum
from django.conf import settings

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
    strategy_description = 'Instantiates and configures a Transformer neural network model for time series processing. Takes architectural parameters including num_layers, d_model, num_heads, d_ff, encoder_input_dim, and decoder_input_dim from param_config, creates a Transformer model instance with these specifications, initializes empty training and validation loss tracking lists, stores the model in entity attributes, and prepares the ModelStageEntity for subsequent training workflows. Essential first step in machine learning pipelines that establishes the neural network architecture before data loading and training.'

    def apply(self, entity: ModelStageEntity):
        param_config = self.strategy_request.param_config
        model = Transformer(param_config)
        
        entity.set_attribute("model", model)
        entity.set_attribute("train_loss", [])
        entity.set_attribute("val_loss", [])

        self.strategy_request.ret_val['entity'] = entity

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

        if not entity.has_attribute('train_loss'):
            entity.set_attribute('train_loss', [])
        if not entity.has_attribute('val_loss'):
            entity.set_attribute('val_loss', [])

        for epoch in range(epochs):
            train_loss = self._train_epoch(model, train_dataloader, criterion, optimizer, device, clip_value)
            val_loss = self._evaluate(model, val_dataloader, criterion, device)
            entity.get_attribute('train_loss').append(train_loss)
            entity.get_attribute('val_loss').append(val_loss)
            self.entity_service.save_entity(entity)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        entity.set_attribute('gradients', self.get_gradients_with_names(model))

        self.strategy_request.ret_val['status'] = 'model_fit_completed'
        self.entity_service.save_entity(entity)
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
            encoder_input, y_target = [x.to(device) for x in batch]

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
                encoder_input, y_target = [x.to(device) for x in batch]
                encoder_input = encoder_input.to(device)
                y_target = y_target.to(device)

                predictions = model(encoder_input)
                loss = criterion(predictions, y_target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def get_gradients_with_names(self, model):
        gradients_with_names = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients_with_names.append(param.grad.norm().item())
        return gradients_with_names


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
        entity.set_attribute('val_loss', val_loss)
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

#TODO The evaluate model strategy should take a prediction input and return the loss no need to forward pass twice

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
        self.entity_service.save_entity(entity)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            "prediction_input_from_entity_name": 'val_dataloader'
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

        batch_size = config.get('batch_size', 64)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        entity.set_attribute('train_dataloader', train_dataloader)
        entity.set_attribute('val_dataloader', val_dataloader)

        entity.set_attribute('device', config['device'])

        self.strategy_request.ret_val['status'] = 'model_configured'
        self.strategy_request.ret_val['entity'] = entity
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
            'device': 'mps',
            'learning_rate': 0.001,
            'batch_size': 64
        }

    def _create_optimizer(self, optimizer_str: str, model):
        optimizer_str = optimizer_str.lower()
        if optimizer_str == OptimizerEnum.OPTIMIZER_ADAM.value:
            return optim.Adam(model.parameters(), lr=self.strategy_request.param_config.get('learning_rate', 0.001))
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_str}")

    def _create_criterion(self, criterion_str: str):
        criterion_str = criterion_str.lower()
        if criterion_str == CriterionEnum.CRITERION_MSE.value:
            return nn.MSELoss()
        if criterion_str == CriterionEnum.MIN_SEQ.value:
            return MinOfNSequenceLoss()
        if criterion_str == CriterionEnum.EXP_MSE.value:
            return MSLELoss()
        if criterion_str == CriterionEnum.GUISS.value:
            return GuissLoss()
        if criterion_str == CriterionEnum.SoftPL.value:
            return SoftInverseProfitLoss()
        if criterion_str == CriterionEnum.SEQ_GUISS.value:
            return SequenceNLLLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_str}")
        


class ComparePredictionsStrategy(ModelStageStrategy):
    """
    Strategy that takes two attributes specified in the param_config:
    1) 'predicted_attribute_name' - The entity attribute holding predicted values of shape (batch, time_steps, 1).
    2) 'actual_attribute_name'    - The entity attribute holding actual values of shape (batch, time_steps, 1).

    It computes the element-wise difference: (predicted - actual)
    and stores the result in a new entity attribute: 'prediction_difference'.
    """
    strategy_description = 'Performs element-wise comparison between model predictions and ground truth values for analysis and evaluation. Takes predicted_attribute_name and actual_attribute_name from param_config to identify entity attributes containing prediction and actual arrays, validates both arrays have matching shapes (typically batch, time_steps, 1), computes the element-wise difference (predicted - actual), and stores the resulting difference array in the entity under "prediction_difference" attribute. Provides essential functionality for error analysis, model performance assessment, and identifying prediction biases across different samples and time steps.'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        config = strategy_request.param_config
        
        # Ensure the param_config has the required keys
        if 'predicted_attribute_name' not in config:
            raise ValueError("param_config must include 'predicted_attribute_name'.")
        if 'actual_attribute_name' not in config:
            raise ValueError("param_config must include 'actual_attribute_name'.")

        # Check if the entity actually has these attributes
        predicted_attr = config['predicted_attribute_name']
        actual_attr = config['actual_attribute_name']

        if not entity.has_attribute(predicted_attr):
            raise ValueError(f"Entity does not have attribute: {predicted_attr}")
        if not entity.has_attribute(actual_attr):
            raise ValueError(f"Entity does not have attribute: {actual_attr}")

    def apply(self, entity: ModelStageEntity):
        # Extract the param_config
        config = self.strategy_request.param_config
        predicted_attr = config['predicted_attribute_name']
        actual_attr = config['actual_attribute_name']

        # Retrieve the predicted and actual arrays/tensors from the entity
        predicted = entity.get_attribute(predicted_attr)
        actual = entity.get_attribute(actual_attr)

        # Optional shape check (assuming (batch, time_steps, 1))
        if predicted.shape != actual.shape:
            raise ValueError(
                f"Predicted and actual must have the same shape. "
                f"Got {predicted.shape} vs {actual.shape}"
            )

        # Compute the difference
        difference = predicted - actual

        # Store the difference in the entity
        entity.set_attribute("prediction_difference", difference)

        # Return the StrategyRequestEntity to signal completion
        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Returns default config keys for this strategy. 
        The user can override 'predicted_attribute_name' and 'actual_attribute_name' 
        with the correct entity keys containing the predicted and actual values.
        """
        return {
            'strategy_name': ComparePredictionsStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'predicted_attribute_name': 'predictions',
                'actual_attribute_name': 'y_test_scaled'
            }
        }

class SaveModelWeightsStrategy(ModelStageStrategy):
    """
    Strategy that saves the weights of the model to a file.
    """
    strategy_description = 'Persists trained model weights to disk for later reuse and deployment. Takes save_name from param_config to generate file path, retrieves the trained model from entity attributes, uses torch.save() to serialize model.state_dict() to {BASE_DIR}/saved_models/{save_name}.pt, stores the complete file path back in entity model_path attribute for future loading operations. Critical for model persistence, enabling trained models to be reloaded without retraining, supporting model versioning, and facilitating deployment workflows.'
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('model'):
            raise ValueError('Model not found in entity.')
        if 'save_name' not in strategy_request.param_config:
            raise ValueError('Save path not found in strategy request.')

    def apply(self, entity: ModelStageEntity):
        model = entity.get_attribute('model')
        save_name = self.strategy_request.param_config['save_name']
        save_path = f"{settings.BASE_DIR}/saved_models/{save_name}.pt"
        torch.save(model.state_dict(), save_path)
        entity.set_attribute('model_path', save_path)
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'save_name': "Model1"
        }

class LoadModelWeightsStrategy(ModelStageStrategy):
    """
    Strategy that loads the weights of the model from a file.
    """
    strategy_description = 'Restores previously saved model weights from disk to resume training or enable inference. Retrieves model_path from entity attributes containing the file location of saved weights, loads the state dictionary using torch.load() from the specified path, applies the loaded weights to the existing model using model.load_state_dict(), effectively restoring the model to its previously trained state. Essential for continuing training from checkpoints, deploying pre-trained models, and implementing model versioning workflows where models need to be restored from persistent storage.'
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('model'):
            raise ValueError('Model not found in entity.')
        if not entity.has_attribute('model_path'):
            raise ValueError('Load path not found in strategy request.')

    def apply(self, entity: ModelStageEntity):
        model = entity.get_attribute('model')
        load_path = entity.get_attribute('model_path')
        model.load_state_dict(torch.load(load_path))
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
        }


class ConfigureDQNModel(ModelStageStrategy):
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('close'):
            raise ValueError('Close price not found in strategy request.')
        if not strategy_request.param_config.get('buffer_size'):
            raise ValueError('Buffer size not found in strategy request.')
        if not strategy_request.param_config.get('device'):
            raise ValueError('Device not found in strategy request.')
        if not strategy_request.param_config.get('num_layers'):
            raise ValueError('num_layers not found in strategy request.')
        if not strategy_request.param_config.get('d_model'):
            raise ValueError('d_model not found in strategy request.')
        if not strategy_request.param_config.get('num_heads'):
            raise ValueError('num_heads not found in strategy request.')
        if not strategy_request.param_config.get('d_ff'):
            raise ValueError('d_ff not found in strategy request.')
        if not strategy_request.param_config.get('dropout'):
            raise ValueError('dropout not found in strategy request.')
        if not strategy_request.param_config.get('encoder_input_dim'):
            raise ValueError('encoder_input_dim not found in strategy request.')

    def apply(self, entity: ModelStageEntity):
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config

        entity.set_attribute("trading_env", TradingEnv(entity.get_attribute('train_dataloader'), entity.get_attribute("close")))

        entity.set_attribute("replay_buffer", ReplayBuffer(config['buffer_size']))

        entity.set_attribute("n_actions", 3)
        entity.set_attribute("num_episodes", 30)
        entity.set_attribute("batch_size", 32)
        entity.set_attribute("gamma", 0.99)
        entity.set_attribute("device", config['device'])

        model_config = {
            "num_layers": config.get('num_layers', 2),
            "d_model": config.get('d_model', 128),
            "num_heads": config.get('num_heads', 8),
            "d_ff": config.get('d_ff', 256),
            "dropout": config.get('dropout', 0.1),
            "encoder_input_dim": config.get('encoder_input_dim', 6),
        }

        policy_net = DQN(model_config).to(config['device'])
        target_net = DQN(model_config).to(config['device'])
        target_net.load_state_dict(policy_net.state_dict())

        entity.set_attribute("optimizer", optim.Adam(policy_net.parameters(), lr=0.001))
        entity.set_attribute("criterion", nn.MSELoss())
        entity.set_attribute("policy_net", policy_net)
        entity.set_attribute("target_net", target_net)
        entity.set_attribute("steps_done", 0)



        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'buffer_size': 10000,
            'num_layers': 2,
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 256,
            'dropout': 0.1,
            'encoder_input_dim': 15,
            'device': 'mps'
        }

class TrainDQNModel(ModelStageStrategy):
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        if not entity.has_attribute('trading_env'):
            raise ValueError('Trading environment not found in entity.')
        if not entity.has_attribute('replay_buffer'):
            raise ValueError('Replay buffer not found in entity.')
        if not entity.has_attribute('policy_net'):
            raise ValueError('Policy network not found in entity.')
        if not entity.has_attribute('target_net'):
            raise ValueError('Target network not found in entity.')
        if not entity.has_attribute('optimizer'):
            raise ValueError('Optimizer not found in entity.')
        if not entity.has_attribute('criterion'):
            raise ValueError('Criterion not found in entity.')
        if not entity.has_attribute('device'):
            raise ValueError('Device not found in entity.')
        if not entity.has_attribute('num_episodes'):
            raise ValueError('Number of episodes not found in entity.')
        if not entity.has_attribute('batch_size'):
            raise ValueError('Batch size not found in entity.')

    def apply(self, entity: ModelStageEntity):
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config

        trading_env = entity.get_attribute('trading_env')
        replay_buffer = entity.get_attribute('replay_buffer')
        policy_net = entity.get_attribute('policy_net')
        target_net = entity.get_attribute('target_net')
        optimizer = entity.get_attribute('optimizer')
        criterion = entity.get_attribute('criterion')
        device = entity.get_attribute('device')

        num_episodes = entity.get_attribute('num_episodes')
        batch_size = entity.get_attribute('batch_size')

        X_train = entity.get_attribute('X_train')

        frame_idx = 0
        for episode in range(num_episodes):
            state = trading_env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            episode_reward = 0
            done = False
            while not done:
                epsilon = epsilon_by_frame(frame_idx)
                frame_idx += 1
                if np.random.rand() < epsilon:
                    action = np.random.choice([0, 1, 2])
                else:
                    with torch.no_grad():
                        q_values = policy_net(state)
                        action = q_values.argmax().item()
                next_state, reward, done, _ = trading_env.step(action)

                episode_reward += reward

                if next_state is not None:
                    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

