from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class InputEntity(Entity):
    entity_name = EntityEnum.INPUT
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)

    def serialize(self) -> dict:
        parent_dict = super().serialize()
        return parent_dict
