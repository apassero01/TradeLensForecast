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
