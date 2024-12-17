from django.test import TestCase
from unittest.mock import patch, MagicMock

from shared_utils.entities import StrategyRequestEntity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from shared_utils.models import StrategyRequest
from training_session.services.TrainingSessionEntityService import TrainingSessionEntityService
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity, TrainingSessionStatus
from training_session.models import TrainingSession
from shared_utils.strategy.BaseStrategy import Strategy

class ExampleStrategy(Strategy):
    def apply(self, entity):
        # Just mark that this strategy was executed
        self.strategy_request.param_config["strategy_executed"] = True
        return self.strategy_request

class TrainingSessionEntityServiceTestCase(TestCase):
    def setUp(self):
        # Mock the StrategyExecutor and StrategyExecutorService since the real code likely needs
        # to interact with external components or more complex logic.
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor.register_strategy("ExampleStrategy", ExampleStrategy)
        self.strategy_executor_service = StrategyExecutorService(self.strategy_executor)

        # Initialize the entity service
        self.entity_service = TrainingSessionEntityService()
        self.entity_service.strategy_executor = self.strategy_executor
        self.entity_service.strategy_executor_service = self.strategy_executor_service

        # Create a training session entity
        self.session_entity = self.entity_service.create_training_session_entity()
        self.session_id = self.session_entity.id

    def test_create_training_session_entity(self):
        """Test that a training session entity is created with default values."""
        self.assertIsInstance(self.session_entity, TrainingSessionEntity)
        self.assertIsNotNone(self.session_entity.entity_id)
        self.assertEqual(self.session_entity.status, TrainingSessionStatus.ACTIVE)
        self.assertEqual(len(self.session_entity.strategy_history), 0)

    def test_get_session(self):
        """Test that we can retrieve a previously created session."""
        retrieved_session = self.entity_service.get_session(self.session_id)
        self.assertEqual(retrieved_session.id, self.session_entity.id)
        self.assertEqual(retrieved_session.entity_id, self.session_entity.entity_id)

    def test_execute_strat_request(self):
        """Test executing a strategy request on the session entity."""
        # Create a strategy request model
        strat_request = StrategyRequest.objects.create(
            strategy_name="ExampleStrategy",
            param_config={"test_param": "value"}
        )

        # Execute the strategy request
        self.entity_service.execute_strat_request(strat_request, self.session_entity)

        # Verify that the strategy was executed and added to the session history
        self.assertTrue(strat_request.param_config.get("strategy_executed"))
        self.assertIn(strat_request, self.session_entity.strategy_history)

    def test_serialize_session(self):
        """Test that we can serialize the session state."""
        strategy_request_entity = StrategyRequestEntity()
        strategy_request_entity.strategy_name = "ExampleStrategy"
        strategy_request_entity.param_config = {"test_param": "value"}


        self.entity_service.execute_strat_request(strategy_request_entity, self.session_entity)

        serialized = self.entity_service.serialize_session()
        self.assertIn('id', serialized)
        self.assertIn('created_at', serialized)
        self.assertIn('entity_map', serialized)
        self.assertIn('strategy_history', serialized)

        # Check that strategy history serialized
        self.assertEqual(len(serialized['strategy_history']), 1)
        self.assertEqual(serialized['strategy_history'][0]['strategy_name'], 'ExampleStrategy')
        self.assertTrue(serialized['strategy_history'][0]['param_config'].get("strategy_executed"))

    def test_save_session(self):
        """Test that we can save the session to the DB."""
        # Modify the session entity in some way
        self.session_entity.set_attribute('some_attr', 'some_value')

        # Save the session
        session_id = self.entity_service.save_session()

        # Retrieve the model and verify
        model = TrainingSession.objects.get(id=session_id)
        self.assertIsNotNone(model)
        self.assertEqual(model.entity_id, self.session_entity.entity_id)

    def test_no_strategy_execution(self):
        """Test that no strategy is executed if we don't add any requests."""
        # Just serialize the session without any strategy execution
        serialized = self.entity_service.serialize_session()
        self.assertEqual(len(serialized['strategy_history']), 0)
