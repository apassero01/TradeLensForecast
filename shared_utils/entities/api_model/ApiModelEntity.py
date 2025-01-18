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
        # Don't include sensitive info like API key in serialization
        parent_dict['meta_data'] = {
            'model_type': self.get_attribute('model_type'),
            'model_name': self.get_attribute('model_name'),
            'has_api_key': self.get_attribute('api_key') is not None
        }
        return parent_dict 