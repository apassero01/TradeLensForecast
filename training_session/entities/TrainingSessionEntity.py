from enum import Enum

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from training_session.models import TrainingSession
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from typing import Optional
class TrainingSessionStatus(Enum):
    ACTIVE = 1
    INACTIVE = 2

class TrainingSessionEntity(Entity):
    entity_name = EntityEnum.TRAINING_SESSION
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.status = TrainingSessionStatus.ACTIVE
        self.created_at = None
        self.strategy_history = []

    def add_to_strategy_history(self, strategy_request):
        self.strategy_history.append(strategy_request)

    @staticmethod
    def from_db(model: TrainingSession):
        """
        Creates a TrainingSessionEntity from a TrainingSession Django model instance
        """
        entity = TrainingSessionEntity()
        entity.id = model.id
        entity.entity_id = model.entity_id
        entity.created_at = model.created_at
        entity.strategy_history = [StrategyRequestEntity.from_db(strategy_request) for strategy_request in model.strategy_history.all()]
        return entity

    @staticmethod
    def entity_to_model(entity, model = None) -> TrainingSession:
        if model is None:
            print(f'entity.id: {entity.id}')
            if entity.id is not None:
                print(f'Getting model with id: {entity.id}')
                model = TrainingSession.objects.get(id=entity.id)
            else:
                model = TrainingSession()

        model.created_at = entity.created_at
        model.strategy_history.set([strategy_request.to_db() for strategy_request in entity.strategy_history])
        model.entity_id = entity.entity_id
        return model

