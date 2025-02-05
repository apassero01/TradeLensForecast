from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class InputEntity(Entity):
    entity_name = EntityEnum.INPUT
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        print('InputEntity init')
        self.set_attribute('data', {})
        self.set_attribute('input_type', None)
        self.set_attribute('config', {})
        self.set_attribute('input', {})

    def set_data(self, data: dict):
        self.set_attribute('data', data)

    def get_data(self) -> dict:
        return self.get_attribute('data')

    def set_input_type(self, input_type: str):
        self.set_attribute('input_type', input_type)

    def get_input_type(self) -> str:
        return self.get_attribute('input_type')

    def set_config(self, config: dict):
        self.set_attribute('config', config)

    def get_config(self) -> dict:
        return self.get_attribute('config')
    
    def serialize(self) -> dict:
        parent_dict = super().serialize()
        parent_dict ['input'] = self.get_attribute('input')
        return parent_dict
