from xxlimited import Error

from django.test import TestCase

from shared_utils.entities.Entity import Entity
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor

class Entity1(Entity):
    entity_name = "Entity1"

    def __init__(self):
        self.entity_map = {"Entity2": None}

class Entity2(Entity):
    entity_name = "Entity2"

    def __init__(self):
        self.entity_map = {"value": "default"}

class Strategy1(Strategy):
    def apply(self, entity):
        # Access Entity2 from Entity1's entity_map
        entity2 = entity.get_entity("Entity2")
        if entity2:
            # Call Strategy2 on Entity2
            nested_requests = self.strategy_request.get_nested_requests()
            nested_request = [r for r in nested_requests if r.strategy_name == "Strategy2"][0]
            nested_requests.remove(nested_request)
            nested_request = self.strategy_executor.execute(entity2, nested_request)
            self.strategy_request.add_nested_request(nested_request)

        # Add execution status to strategy_request
        self.strategy_request.param_config["strategy1_executed"] = True
        return self.strategy_request

class Strategy2(Strategy):
    def apply(self, entity):
        # Add execution status to strategy_request
        self.strategy_request.param_config["strategy2_executed"] = True
        return self.strategy_request


class StrategyExecutorTestCase(TestCase):
    def setUp(self):
        # Initialize the StrategyExecutor and register strategies
        self.executor = StrategyExecutor()
        self.executor.register_strategy("Strategy1", Strategy1)
        self.executor.register_strategy("Strategy2", Strategy2)

        # Create test entities
        self.entity2 = Entity2()
        self.entity1 = Entity1()
        self.entity1.set_entity_map({"Entity2": self.entity2})

        # Create a mock StrategyRequest
        self.strategy_request1 = StrategyRequest.objects.create(param_config={"test_param": "value"})
        self.strategy_request1.strategy_name = "Strategy1"
        self.strategy_request2 = StrategyRequest.objects.create(param_config={"test_param": "value"})
        self.strategy_request2.strategy_name = "Strategy2"

        self.strategy_request1.save()
        self.strategy_request2.save()


    def test_singleton_behavior(self):
        # Test that only one instance of StrategyExecutor can be created
        executor1 = StrategyExecutor()
        executor2 = StrategyExecutor()
        self.assertEqual(executor1, executor2)

    def test_recreation_of_executor(self):
        # Test that the executor can be recreated after being destroyed
        executor1 = StrategyExecutor()
        StrategyExecutor.destroy()  # Explicitly destroy the instance

        # Now try to create another instance
        executor2 = StrategyExecutor()
        self.assertNotEqual(executor1, executor2)
        self.assertTrue(executor2 is not None)

    def test_single_strategy_execution(self):
        """Test executing a single strategy on Entity2."""

        strategy_request = self.executor.execute(self.entity2, self.strategy_request2)

        # Verify that Strategy2 was executed
        self.assertTrue(strategy_request.param_config.get("strategy2_executed"))

    def test_nested_strategy_execution(self):
        """Test executing Strategy1, which calls Strategy2."""
        self.strategy_request1.add_nested_request(self.strategy_request2)
        strategy_request = self.executor.execute(self.entity1, self.strategy_request1)

        # Verify that both Strategy1 and Strategy2 were executed
        self.assertTrue(strategy_request.param_config.get("strategy1_executed"))
        strategy_request2 = strategy_request.get_nested_requests()[0]
        self.assertTrue(strategy_request2.param_config.get("strategy2_executed"))

    def test_multiple_top_level_executions(self):
        """Test multiple independent top-level strategy executions."""
        strategy_request2_1 = self.executor.execute(self.entity2, self.strategy_request2)
        strategy_request2 = self.executor.execute(self.entity2, self.strategy_request2)

        # Verify both strategies executed independently
        self.assertTrue(strategy_request2_1.param_config.get("strategy2_executed"))
        self.assertTrue(strategy_request2.param_config.get("strategy2_executed"))

    def test_strategy_request_return_dict(self):
        """Test that the strategy request correctly tracks execution statuses."""
        self.strategy_request1.add_nested_request(self.strategy_request2)
        strategy_request = self.executor.execute(self.entity1, self.strategy_request1)

        # Ensure both strategies updated the strategy request
        self.assertTrue(strategy_request.param_config.get("strategy1_executed"))
        strategy_request2 = strategy_request.get_nested_requests()[0]
        self.assertTrue(strategy_request2.param_config.get("strategy2_executed"))

    def test_strategy_execution_error_propagation(self):
        """Test that an error in a nested strategy execution propagates up the stack."""
        # Define a failing strategy that raises an exception
        class FailingStrategy(Strategy):
            def apply(self, entity):
                raise ValueError("Intentional failure in FailingStrategy")

        # Register the failing strategy
        self.executor.register_strategy("FailingStrategy", FailingStrategy)

        # Create a new strategy request for the failing strategy
        failing_strategy_request = StrategyRequest.objects.create(param_config={"test_param": "value"})
        failing_strategy_request.strategy_name = "FailingStrategy"


        # Expect an error when executing Strategy1
        with self.assertRaises(ValueError) as context:
            self.executor.execute(self.entity1, failing_strategy_request)

        # Verify the exception message
        self.assertEqual(str(context.exception), "Intentional failure in FailingStrategy")

        # Ensure that Strategy1 did not complete its execution due to the nested failure
        self.assertFalse(self.strategy_request1.param_config.get("strategy1_executed", False))