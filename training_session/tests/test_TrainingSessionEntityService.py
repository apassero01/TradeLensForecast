from django.test import TestCase
from shared_utils.entities.Entity import Entity

from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.models import StrategyRequest
from train_eval_manager.entities.entities import ModelEntity, ModelStageEntity
from training_session.entities.entities import TrainingSessionEntity
from training_session.services.TrainingSessionEntityService import TrainingSessionEntityService


class ExampleStrategy(Strategy):
    def apply(self, entity):
        # Mark the strategy as executed
        self.strategy_request.param_config["strategy_executed"] = True
        return self.strategy_request


class TrainingSessionEntityServiceTestCase(TestCase):
    def setUp(self):
        # Initialize StrategyExecutor and register a simple strategy
        self.executor = StrategyExecutor()
        self.executor.register_strategy("ExampleStrategy", ExampleStrategy)

        # Initialize the entity service
        self.entity_service = TrainingSessionEntityService()

        # Create entities
        self.model_entity = ModelEntity(model="Model1")
        self.model_stage_entity = ModelStageEntity(model_stage="ModelStage1")
        self.training_session_entity = TrainingSessionEntity(session="Session1")

        # Set up the entity map hierarchy
        self.model_stage_entity.set_entity_map({"model": self.model_entity})
        self.training_session_entity.set_entity_map({"model_stage": self.model_stage_entity})

        # Create a StrategyRequest object
        self.strategy_request = StrategyRequest.objects.create(
            strategy_name="ExampleStrategy",
            strategy_path="training_session.model_stage.model",
            param_config={"test_param": "value", "path": "training_session.model_stage.model"}
        )

    def test_resolve_strat_request_path(self):
        """Test resolving a nested path."""
        resolved_entity = self.entity_service.resolve_strat_request_path(
            self.strategy_request,
            self.training_session_entity
        )

        # Ensure the resolved entity is the model entity
        self.assertEqual(resolved_entity, self.model_entity)

    def test_execute_strat_request(self):
        """Test executing a strategy on a nested entity."""
        # Execute the strategy
        self.entity_service.execute_strat_request(
            self.strategy_request, self.training_session_entity
        )

        # Verify that the strategy was executed
        self.assertTrue(self.strategy_request.param_config.get("strategy_executed"))

    def test_invalid_path(self):
        """Test that an invalid path raises an error."""
        with self.assertRaises(ValueError) as context:
            self.strategy_request.strategy_path = "training_session.invalid_entity"
            self.entity_service.resolve_strat_request_path(
                self.strategy_request,
                self.training_session_entity
            )

        self.assertEqual(str(context.exception), "Key invalid_entity not found in entity map")