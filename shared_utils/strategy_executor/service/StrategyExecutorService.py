from data_bundle_manager.strategy.DataBundleStrategy import LoadDataBundleDataStrategy, CreateFeatureSetsStrategy, \
    SplitBundleDateStrategy, ScaleByFeatureSetsStrategy, CombineDataBundlesStrategy
from sequenceset_manager.strategy.SequenceSetStrategy import CreateDataBundleStrategy, PopulateDataBundleStrategy, \
    SplitAllBundlesDataStrategy, ScaleSeqSetsByFeaturesStrategy, CombineSeqBundlesStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor import StrategyExecutor
from train_eval_manager.strategies.ModelStageStrategy import CreateModelStrategy
from training_session.strategy.TrainingSessionStrategy import GetSequenceSetsStrategy, CreateModelStageStrategy, \
    GetDataBundleStrategy


class StrategyExecutorService:
    registry = {
        EntityEnum.TRAINING_SESSION.value: [
            GetSequenceSetsStrategy,
            CreateModelStageStrategy,
            GetDataBundleStrategy
        ],
        EntityEnum.DATA_BUNDLE.value: [
            LoadDataBundleDataStrategy,
            CreateFeatureSetsStrategy,
            SplitBundleDateStrategy,
            ScaleByFeatureSetsStrategy,
            CombineDataBundlesStrategy
        ],
        EntityEnum.MODEL.value: [],
        EntityEnum.MODEL_STAGE.value: [
            CreateModelStrategy
        ],
        EntityEnum.FEATURE_SETS.value: [

        ],
        EntityEnum.SEQUENCE_SETS.value: [
            CreateDataBundleStrategy,
            PopulateDataBundleStrategy,
            SplitAllBundlesDataStrategy,
            ScaleSeqSetsByFeaturesStrategy,
            CombineSeqBundlesStrategy

        ],
        EntityEnum.STRATEGY_REQUESTS.value: [],

    }

    def __init__(self, strategy_executor: StrategyExecutor):
        self.strategy_executor = strategy_executor
        self.register_strategies()


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

