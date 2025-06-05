from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional, Dict, Any

class ApiModelEntity(Entity):
    entity_name = EntityEnum.API_MODEL

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute('model_type', None)  # e.g., 'openai', 'anthropic', etc
        self.set_attribute('model_name', None)  # e.g., 'gpt-4', 'claude-2', etc
        self.set_attribute('api_key', None)
        self.set_attribute('config', {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 1.0
        })
        self.set_attribute('last_response', None)
        self.set_attribute('message_history', [])

    def serialize(self) -> dict:
        parent_dict = super().serialize()
        parent_dict['model_type'] = self.get_attribute('model_type')
        parent_dict['model_name'] = self.get_attribute('model_name')
        parent_dict['message_history'] = [{'type': message.type, 'content': message.content} for message in self.get_attribute('message_history')]
        return parent_dict 