from unittest.mock import MagicMock
from django.test import TestCase

from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity


class TestEntity(Entity):
    entity_name = EntityEnum.ENTITY

    def to_db(self):
        return {}

    @classmethod
    def from_db(cls, data):
        return cls()


class Strategy1(Strategy):
    def apply(self, entity):
        # Execute the nested strategy on the same entity
        nested_requests = self.strategy_request.get_nested_requests()
        if nested_requests:
            nested_request = nested_requests.pop(0)
            nested_request = self.strategy_executor.execute(entity, nested_request)
            self.strategy_request.add_nested_request(nested_request)

        # Mark the strategy as executed
        self.strategy_request.param_config["strategy1_executed"] = True
        return self.strategy_request


class Strategy2(Strategy):
    def apply(self, entity):
        # Mark the strategy as executed
        self.strategy_request.param_config["strategy2_executed"] = True
        return self.strategy_request


class StrategyExecutorTestCase(TestCase):
    def setUp(self):
        # Initialize the StrategyExecutor and register strategies
        self.executor = StrategyExecutor()
        self.executor.register_strategy("Strategy1", Strategy1)
        self.executor.register_strategy("Strategy2", Strategy2)

        # Create a test entity
        self.entity = TestEntity()
        self.entity.entity_id = "test_entity"

        # Create strategy requests
        self.strategy_request1 = StrategyRequestEntity()
        self.strategy_request1.strategy_name = "Strategy1"
        self.strategy_request1.param_config = {"test_param": "value"}

        self.strategy_request2 = StrategyRequestEntity()
        self.strategy_request2.strategy_name = "Strategy2"
        self.strategy_request2.param_config = {"test_param": "value"}

    def test_singleton_behavior(self):
        """Test that only one instance of StrategyExecutor can be created"""
        executor1 = StrategyExecutor()
        executor2 = StrategyExecutor()
        self.assertEqual(executor1, executor2)

    def test_recreation_of_executor(self):
        """Test that the executor can be recreated after being destroyed"""
        executor1 = StrategyExecutor()
        StrategyExecutor.destroy()
        executor2 = StrategyExecutor()
        self.assertNotEqual(executor1, executor2)

    def test_single_strategy_execution(self):
        """Test executing a single strategy on the entity"""
        strategy_request = self.executor.execute(self.entity, self.strategy_request2)
        self.assertTrue(strategy_request.param_config.get("strategy2_executed"))

    def test_nested_strategy_execution(self):
        """Test executing Strategy1, which calls Strategy2 on the same entity"""
        self.strategy_request1.add_nested_request(self.strategy_request2)
        strategy_request = self.executor.execute(self.entity, self.strategy_request1)

        # Verify both strategies were executed
        self.assertTrue(strategy_request.param_config.get("strategy1_executed"))
        nested_request = strategy_request.get_nested_requests()[0]
        self.assertTrue(nested_request.param_config.get("strategy2_executed"))

    def test_multiple_top_level_executions(self):
        """Test multiple independent top-level strategy executions"""
        strategy_request1 = self.executor.execute(self.entity, self.strategy_request1)
        strategy_request2 = self.executor.execute(self.entity, self.strategy_request2)

        # Verify both strategies executed independently
        self.assertTrue(strategy_request1.param_config.get("strategy1_executed"))
        self.assertTrue(strategy_request2.param_config.get("strategy2_executed"))