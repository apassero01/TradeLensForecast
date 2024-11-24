from abc import ABC, abstractmethod

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor


class Strategy(ABC):
    entity_type = EntityEnum.ENTITY
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    @abstractmethod
    def apply(self, entity):
        """Apply the strategy to the entity."""
        pass

    @abstractmethod
    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}

    @classmethod
    def serialize(cls):
        return {
            'name': cls.__name__,
            'config': cls.get_request_config()
        }

