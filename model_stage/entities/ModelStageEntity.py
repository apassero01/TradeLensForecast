from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class ModelStageEntity(Entity):
    entity_name = EntityEnum.MODEL_STAGE
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)

    def serialize(self):
        return {
            'entity_name': self.entity_name.value,
            'path': self.path,
            'class_path': self.__class__.__module__ + '.' + self.__class__.__name__,
            'children': [child.serialize() for child in self.children],
            'meta_data': {
                # 'model': self.get_attribute("model").config if self.has_attribute("model") else None,
                'optimizer': self.get_attribute("optimizer_name") if self.has_attribute("optimizer_name") else None,
                'criterion': self.get_attribute("criterion_name") if self.has_attribute("criterion_name") else None

            }
        }


        
