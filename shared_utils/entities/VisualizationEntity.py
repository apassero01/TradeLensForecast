from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional


class VisualizationEntity(Entity):
    entity_name = EntityEnum.VISUALIZATION
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        print('VisualizationEntity init')
        self.set_attribute('data', {})
        self.set_attribute('visualization_type', None)
        self.set_attribute('config', {})
        self.set_attribute('visualization', {})

    def set_data(self, data: dict):
        self.set_attribute('data', data)

    def get_data(self) -> dict:
        return self.get_attribute('data')

    def set_visualization_type(self, viz_type: str):
        self.set_attribute('visualization_type', viz_type)

    def get_visualization_type(self) -> str:
        return self.get_attribute('visualization_type')
    
    def set_config(self, config: dict):
        self.set_attribute('config', config)

    def get_config(self) -> dict:
        return self.get_attribute('config')
    
    def serialize(self) -> dict:
        return {
            'entity_name': self.entity_name.value,
            'path': self.path,
            'visualization': self.get_attribute('visualization')
        }
