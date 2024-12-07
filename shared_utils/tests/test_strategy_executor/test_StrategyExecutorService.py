from django.test import TestCase
from unittest.mock import MagicMock, patch
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import Strategy

class TestStrategy(Strategy):
    def apply(self, entity):
        # Mark the strategy as executed
        self.strategy_request.param_config["strategy_executed"] = True
        return self.strategy_request

class TestStrategyExecutorService(TestCase):
    def setUp(self):
        # Create a mock executor instead of real one
        self.executor = MagicMock(spec=StrategyExecutor)
        
        # Configure mock to return strategy request with executed flag
        def mock_execute(entity, strategy_request):
            strategy_request.param_config["strategy_executed"] = True
            return strategy_request
        self.executor.execute.side_effect = mock_execute
        
        self.service = StrategyExecutorService(self.executor)
        self.entity = MagicMock()
        self.strategy_request = StrategyRequest.objects.create(
            strategy_name="TestStrategy",
            param_config={"test_param": "value"}
        )

    def test_execute_with_path(self):
        """Test executing a strategy with path resolution"""
        # Setup target entity
        target_entity = MagicMock()
        self.entity.find_entity_by_path.return_value = target_entity
        
        # Set path in strategy request
        self.strategy_request.strategy_path = "path/to/entity"

        # Execute
        result = self.service.execute(self.entity, self.strategy_request)

        # Verify
        self.entity.find_entity_by_path.assert_called_once_with(self.strategy_request.strategy_path)
        self.executor.execute.assert_called_once_with(target_entity, self.strategy_request)
        self.assertTrue(result.param_config["strategy_executed"])

    def test_execute_without_path(self):
        """Test executing a strategy without path resolution"""
        # Execute without path
        result = self.service.execute(self.entity, self.strategy_request)

        # Verify direct execution on entity
        self.entity.find_entity_by_path.assert_not_called()
        self.executor.execute.assert_called_once_with(self.entity, self.strategy_request)
        self.assertTrue(result.param_config["strategy_executed"]) 