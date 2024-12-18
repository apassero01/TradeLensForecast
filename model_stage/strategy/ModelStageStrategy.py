from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.Entity import Entity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from model_stage.entities.ModelStageEntity import ModelStageEntity
from shared_utils.entities.EnityEnum import EntityEnum
from models.BuiltModels import Transformer

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
        model = Transformer(num_layers=param_config['num_layers'], 
                            d_model=param_config['d_model'], 
                            num_heads=param_config['num_heads'], 
                            d_ff=param_config['d_ff'], 
                            encoder_input_dim=param_config['encoder_input_dim'], 
                            decoder_input_dim=param_config['decoder_input_dim'])
        
        entity.set_attribute("model", model)

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




