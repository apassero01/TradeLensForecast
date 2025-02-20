from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class ModelStageEntity(Entity):
    entity_name = EntityEnum.MODEL_STAGE
    db_attributes = ['model_path']
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)

    def serialize(self):
        super_dict = super().serialize()
        super_dict['meta_data'] =  {
                # 'model': self.get_attribute("model").config if self.has_attribute("model") else None,
                'optimizer': self.get_attribute("optimizer_name") if self.has_attribute("optimizer_name") else None,
                'criterion': self.get_attribute("criterion_name") if self.has_attribute("criterion_name") else None,
                'val_loss': self.get_attribute("val_loss")if self.has_attribute("val_loss") else None,
            'predictions': self.get_attribute("predictions").shape if self.has_attribute("predictions") else None,
            'results': self.get_attribute("results").shape if self.has_attribute("results") else None,
        }

        return super_dict


        
