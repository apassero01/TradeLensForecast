from abc import ABC, abstractmethod

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.models import StrategyRequest
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor


class Strategy(ABC):
    entity_type = EntityEnum.ENTITY
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequest):
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    @abstractmethod
    def apply(self, entity):
        """Apply the strategy to the entity."""
        pass

    @staticmethod
    def required_entities():
        return {} # Override this method to specify the required data objects for the strategy

    @staticmethod
    def get_request_config():
        return {}
