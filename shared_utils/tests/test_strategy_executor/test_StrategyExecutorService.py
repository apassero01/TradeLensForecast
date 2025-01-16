import uuid
from django.test import TestCase

from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity

class MockStrategy(Strategy):
    """
    A simple strategy that sets `test_key` on the entity.
    """
    def apply(self, entity):
        entity.set_attribute('test_key', 'test_value')
        return self.strategy_request

class TestStrategyExecutorServiceNoMock(TestCase):
    """
    Full integration-style tests for StrategyExecutorService without mocking.
    Uses the real StrategyDirectory, real EntityService, real StrategyExecutor, etc.
    """

    def setUp(self):
        """
        Runs before each test.
        - Instantiates a real StrategyExecutorService with real StrategyExecutor.
        - Creates an Entity and StrategyRequestEntity referencing that entity.
        - Saves the entity so it can be retrieved by ID during tests.
        """
        # Real service under test
        self.service = StrategyExecutorService(StrategyExecutor())

        # Real entity service
        self.entity_service = EntityService()

        # Create a real Entity
        self.entity = Entity()
        self.entity_service.save_entity(self.entity)

        # Create a request pointing to MockStrategy
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.target_entity_id = self.entity.entity_id
        self.strategy_request.strategy_name = MockStrategy.__name__

        # Register the MockStrategy manually on the strategy executor,
        # since it might not be in your StrategyDirectory by default.
        self.service.strategy_executor.register_strategy(MockStrategy.__name__, MockStrategy)

    def test_execute_request_success(self):
        """
        Ensures that execute_request(...) applies the MockStrategy, which sets
        `test_key` = `test_value` on the entity.
        """
        # Call the method under test
        result_request = self.service.execute_request(self.strategy_request)

        # Reload the entity from the service to see if it was updated & saved
        updated_entity = self.entity_service.get_entity(self.entity.entity_id)

        # Check that the 'test_key' attribute was set by MockStrategy
        self.assertEqual(updated_entity.get_attribute('test_key'), 'test_value')
        # Also ensure the returned request is the same request object
        self.assertEqual(result_request, self.strategy_request)

    def test_execute_request_entity_not_found(self):
        """
        If the target_entity_id doesn't exist, the service should raise ValueError.
        """
        # Create a new request with a random (non-existent) UUID
        bad_request = StrategyRequestEntity()
        bad_request.target_entity_id = str(uuid.uuid4())
        bad_request.strategy_name = MockStrategy.__name__

        with self.assertRaises(ValueError) as ctx:
            self.service.execute_request(bad_request)

        self.assertIn("Entity not found for id:", str(ctx.exception))

    def test_execute(self):
        """
        Verifies calling .execute(...) directly updates the entity and saves it.
        """
        # Directly call execute(...) with the known entity and request
        returned_request = self.service.execute(self.entity, self.strategy_request)

        # Fetch the entity again
        updated_entity = self.entity_service.get_entity(self.entity.entity_id)

        # Confirm that MockStrategy was applied
        self.assertEqual(updated_entity.get_attribute('test_key'), 'test_value')
        self.assertEqual(returned_request, self.strategy_request)

    def test_get_registry(self):
        """
        Verifies get_registry() returns a dictionary of all strategies,
        including the newly-registered MockStrategy if your StrategyDirectory
        or code merges them properly.
        """
        registry = self.service.get_registry()

        # The keys in 'registry' are the entity types from StrategyDirectory (e.g. "entity", "model_stage", etc.).
        # We check it is at least not empty.
        self.assertTrue(isinstance(registry, dict), "Expected registry to be a dict.")
        self.assertTrue(len(registry) > 0, "Expected registry to have at least one entity type key.")





