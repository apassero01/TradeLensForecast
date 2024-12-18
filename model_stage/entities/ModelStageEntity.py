from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class ModelStageEntity(Entity):
    entity_name = EntityEnum.MODEL_STAGE
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        
