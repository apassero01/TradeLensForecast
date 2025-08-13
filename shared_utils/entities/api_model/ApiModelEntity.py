from uuid import uuid4
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional, Dict, Any

from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy

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
        self.set_attribute('visible_entities', [])

    def on_create(self, param_config: Optional[Dict[str, Any]] = None):
        strategy_request_list = []
        child_vis_create  = StrategyRequestEntity()
        self.add_child(child_vis_create)

        child_vis_create.strategy_name = CreateEntityStrategy.__name__
        child_uuid = str(uuid4())
        child_vis_create.param_config = {
            'entity_class': ViewEntity.get_class_path(),
            'entity_uuid': child_uuid,
            'initial_attributes': {
                'view_component_type': 'chatinterface',
            }
        }
        child_vis_create.target_entity_id = self.entity_id
        child_vis_create.add_to_history = False

        strategy_request_list.append(child_vis_create)

        return strategy_request_list


    def serialize(self) -> dict:
        parent_dict = super().serialize()
        parent_dict['model_type'] = self.get_attribute('model_type')
        parent_dict['model_name'] = self.get_attribute('model_name')
        parent_dict['message_history'] = self.serialize_message_history()
        parent_dict['visible_entities'] = self.get_attribute('visible_entities')
        parent_dict['name'] = self.get_attribute('name') if self.has_attribute('name') else None
        return parent_dict

    def serialize_message_history(self) -> list:
        messages = []
        for message in self.get_attribute('message_history'):
            if type(message.content) is list:
                content = ''
                for single_message in message.content:
                    if isinstance(single_message, dict) and 'text' in single_message:
                        content += single_message['text'] + '\n'
                    elif isinstance(single_message, str):
                        content += single_message + '\n'
                    else:
                        # Handle other types by converting to string
                        content += str(single_message) + '\n'
            else :
                content = message.content

            messages.append({
                'type': message.type,
                'content': content
            })
        return messages

