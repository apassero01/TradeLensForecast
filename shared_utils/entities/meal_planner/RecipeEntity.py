from typing import Optional
from uuid import uuid4

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy


class RecipeEntity(Entity):
    entity_name = EntityEnum.RECIPE
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute('name', '')
        self.set_attribute('instructions', '')
        self.set_attribute('ingredients', [])
        self.set_attribute("hidden", True)

    def on_create(self, param_config: Optional[dict] = None) -> list:
        """Override this method to handle entity creation logic should return list of requests to operate"""
        requests = []
        instruction_child_id = str(uuid4())
        instruction_view_attributes = {
            'parent_attributes': {"instructions": "instructions", "ingredients": "ingredients"},
            'view_component_type': 'recipeinstructions',
        }
        instruction_view_request = CreateEntityStrategy.request_constructor(self.entity_id, ViewEntity.get_class_path(), entity_uuid=instruction_child_id, initial_attributes=instruction_view_attributes)
        requests.append(instruction_view_request)

        list_view_child_id = str(uuid4())
        list_view_attributes = {
            'parent_attributes': {"name": "name"},
            'view_component_type': 'recipelistitem',
        }
        list_view_request = CreateEntityStrategy.request_constructor(self.entity_id, ViewEntity.get_class_path(), entity_uuid=list_view_child_id, initial_attributes=list_view_attributes)
        requests.append(list_view_request)

        return requests

    def serialize(self):
        """Override this method to handle entity serialization logic"""
        serialized_data = super().serialize()
        serialized_data['name'] = self.get_attribute('name')
        serialized_data['instructions'] = self.get_attribute('instructions')
        serialized_data['ingredients'] = self.get_attribute('ingredients')
        return serialized_data

