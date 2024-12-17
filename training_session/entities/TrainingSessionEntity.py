from enum import Enum

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from training_session.models import TrainingSession
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity, StrategyRequestAdapter
from shared_utils.models import StrategyRequest
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
    def from_db(model: TrainingSession) -> 'TrainingSessionEntity':
        """Convert a TrainingSession model to a TrainingSessionEntity"""
        entity = TrainingSessionEntity(entity_id=model.entity_id)
        entity.id = model.pk
        entity.created_at = model.created_at

        # Get strategy histories associated with this specific training session
        strategy_requests = StrategyRequest.objects.filter(
            training_session=model, 
            parent_request__isnull=True
        )
        for strategy_request in strategy_requests:
            strategy_entity = StrategyRequestAdapter.model_to_entity(strategy_request)
            entity.strategy_history.append(strategy_entity)
        
        return entity

    @staticmethod
    def to_db(entity: 'TrainingSessionEntity', model: Optional[TrainingSession] = None) -> TrainingSession:
        """Convert a TrainingSessionEntity to a TrainingSession model"""
        if model is None:
            if entity.id is not None:
                model = TrainingSession.objects.get(id=entity.id)
            else:
                model = TrainingSession()

        # Update timestamps
        model.created_at = entity.created_at

        if not model.pk:  # If this is a new model, save it
            model.save()

        # Update strategy histories (convert StrategyRequestEntity to StrategyRequest)
        existing_strategy_ids = set(model.strategy_requests.values_list('id', flat=True))
        for strategy_entity in entity.strategy_history:
            strategy_model = StrategyRequestAdapter.entity_to_model(strategy_entity)
            strategy_model.training_session = model  # Set the link to this training session
            if strategy_model.pk not in existing_strategy_ids:
                strategy_model.save()

        return model

