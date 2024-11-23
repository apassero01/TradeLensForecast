from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy
from train_eval_manager.entities.entities import ModelStageEntity
from training_session.entities.entities import TrainingSessionEntity


class TrainingSessionStrategy(Strategy):
    '''
    The TrainingSessionStrategy class is a concrete class for manipulating data within the TrainingSession.

    TrainingSessionStrategy components can manipulate data within the ModelStage.

    For example a TrainingSessionService could create a feature set inside the TrainingSession.
    '''
    entity_type = EntityEnum.TRAINING_SESSION
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        pass

class CreateModelStageStrategy(TrainingSessionStrategy):
    '''
    The CreateModelStageStrategy class is a concrete class for creating a model stage inside the TrainingSessionEntity.
    '''
    name = "CreateModelStage"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        training_session_entity = session_entity
        model_stage = "test_model_stage"
        model_stage_entity = ModelStageEntity(model_stage)

        training_session_entity.set_entity_map({ModelStageEntity.entity_name: model_stage_entity})

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CreateModelStageStrategy.__name__,
            'strategy_path': 'training_session',
            'param_config': {}
        }




