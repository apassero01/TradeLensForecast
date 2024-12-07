from unittest.mock import MagicMock
from django.test import TestCase

from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity

class TestEntity1(Entity):
    entity_name = EntityEnum.ENTITY

    def to_db(self):
        return {}
        
    @classmethod
    def from_db(cls, data):
        return cls()

class TestEntity2(Entity):
    entity_name = EntityEnum.DATA_BUNDLE

    def to_db(self):
        return {}
        
    @classmethod
    def from_db(cls, data):
        return cls()

class Strategy1(Strategy):
    def apply(self, entity):
        # Find Entity2 in children
        entity2 = next(
            (child for child in entity.children 
             if child.entity_name == EntityEnum.DATA_BUNDLE), 
            None
        )
        
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
        self.entity2 = TestEntity2()
        self.entity1 = TestEntity1()
        self.entity1.add_child(self.entity2)  # Use add_child instead of entity_map

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
        """Test executing a single strategy on Entity2"""
        strategy_request = self.executor.execute(self.entity2, self.strategy_request2)
        self.assertTrue(strategy_request.param_config.get("strategy2_executed"))

    def test_nested_strategy_execution(self):
        """Test executing Strategy1, which calls Strategy2"""
        self.strategy_request1.add_nested_request(self.strategy_request2)
        strategy_request = self.executor.execute(self.entity1, self.strategy_request1)

        # Verify both strategies were executed
        self.assertTrue(strategy_request.param_config.get("strategy1_executed"))
        strategy_request2 = strategy_request.get_nested_requests()[0]
        self.assertTrue(strategy_request2.param_config.get("strategy2_executed"))

    def test_multiple_top_level_executions(self):
        """Test multiple independent top-level strategy executions"""
        strategy_request2_1 = self.executor.execute(self.entity2, self.strategy_request2)
        strategy_request2_2 = self.executor.execute(self.entity2, self.strategy_request2)

        # Verify both strategies executed independently
        self.assertTrue(strategy_request2_1.param_config.get("strategy2_executed"))
        self.assertTrue(strategy_request2_2.param_config.get("strategy2_executed"))

