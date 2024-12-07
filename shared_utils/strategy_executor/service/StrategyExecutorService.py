from data_bundle_manager.strategy.DataBundleStrategy import CreateFeatureSetsStrategy, \
    SplitBundleDateStrategy, ScaleByFeatureSetsStrategy, CombineDataBundlesStrategy
from sequenceset_manager.strategy.SequenceSetStrategy import PopulateDataBundleStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor import StrategyExecutor
from train_eval_manager.strategies.ModelStageStrategy import CreateModelStrategy
from training_session.strategy.TrainingSessionStrategy import GetSequenceSetsStrategy, CreateModelStageStrategy


class StrategyExecutorService:
    registry = {
        EntityEnum.TRAINING_SESSION.value: [
            GetSequenceSetsStrategy,
            CreateModelStageStrategy
        ],
        EntityEnum.DATA_BUNDLE.value: [
            CreateFeatureSetsStrategy,
            SplitBundleDateStrategy,
            ScaleByFeatureSetsStrategy,
            CombineDataBundlesStrategy
        ],
        EntityEnum.MODEL.value: [],
        EntityEnum.MODEL_STAGE.value: [
            CreateModelStrategy
        ],
        EntityEnum.FEATURE_SET.value: [

        ],
        EntityEnum.SEQUENCE_SET.value: [

            PopulateDataBundleStrategy,
        ],
        EntityEnum.STRATEGY_REQUESTS.value: [],

    }

    def __init__(self, strategy_executor: StrategyExecutor):
        self.strategy_executor = strategy_executor
        self.register_strategies()


    def execute(self, entity, strategy_request):
        """Execute a strategy on an entity after resolving its path"""
        if hasattr(strategy_request, 'strategy_path') and strategy_request.strategy_path:
            target_entity = entity.find_entity_by_path(strategy_request.strategy_path)
            if target_entity is None:
                raise ValueError(f'Entity not found for path: {strategy_request.strategy_path}')
            entity = target_entity
        return self.strategy_executor.execute(entity, strategy_request)


    def register_strategies(self):
        for entity, strategies in StrategyExecutorService.registry.items():
            for strategy in strategies:
                self.strategy_executor.register_strategy(strategy.__name__, strategy)


    @staticmethod
    def get_registry():
        serialized_registry = []
        for entity, strategies in StrategyExecutorService.registry.items():
            serialized_registry += [s.serialize() for s in strategies]
        return serialized_registry

