from django.test import TestCase

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.models import StrategyRequest
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from train_eval_manager.entities.entities import ModelEntity, ModelStageEntity
from train_eval_manager.strategies.ModelStageStrategy import CreateModelStrategy


class CreateModelStrategyTestCase(TestCase):
    def setUp(self):
        self.model_stage_data_object = ModelStageEntity(None)
        self.strategy_executor = StrategyExecutor()

    def test_CreateModelStrategy(self):
        strategy_request = StrategyRequest()

        strategy = CreateModelStrategy(self.strategy_executor, strategy_request)
        strategy.apply(self.model_stage_data_object)

        self.assertEqual(type(self.model_stage_data_object.get_entity(EntityEnum.MODEL.value)), ModelEntity)