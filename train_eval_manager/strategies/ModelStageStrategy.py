from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from train_eval_manager.entities.entities import ModelEntity, ModelStageEntity


class ModelStageStrategy(Strategy):
    '''
    The ModelStageStrategy class is a base class for all strategies that manipulate the ModelStageEntity.

    ModelStageStrategy components can manipulate data within the ModelStage.

    For example, a Trainer strategy could create a new model inside the trainer object.
    Or a Trainer strategy could perform a fit operation on the model inside the trainer object.
    '''
    entity_type = EntityEnum.MODEL_STAGE
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_object):
        pass

class CreateModelStrategy(ModelStageStrategy):
    '''
    The CreateModelStrategy class is a concrete class for creating a model inside the ModelStageDataObject.

    '''
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_object):
        model_stage_data_object = data_object
        model_config = self.strategy_request.param_config

        model = "test_model"
        model_object = ModelEntity(model)

        model_stage_data_object.set_entity_map({ModelEntity.entity_name.value: model_object})


