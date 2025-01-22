from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from shared_utils.entities.EnityEnum import EntityEnum
from uuid import uuid4, UUID
import numpy as np
class Entity(ABC):
    entity_name = EntityEnum.ENTITY

    def __init__(self, entity_id: Optional[str] = None):
        # Validate entity_id if provided
        if entity_id is not None:
            try:
                # Verify it's a valid UUID string
                UUID(entity_id)
            except ValueError:
                raise ValueError(f"Invalid UUID format: {entity_id}")
        
        # Generate new UUID if none provided
        self.entity_id = entity_id if entity_id is not None else str(uuid4())
        self._attributes: Dict[str, Any] = {}
        self.children_ids = []
        self.parent_ids = []
        self.strategy_requests = []

    def on_create(self, param_config: Optional[Dict[str, Any]] = None):
        pass

    def add_child(self, child):
        '''Add a child to the entity'''
        self.children_ids.append(child.entity_id)
        child.add_parent(self)
    
    def remove_child(self, child: 'Entity'):
        '''Remove a child from the entity'''
        if child.entity_id in self.children_ids:
            self.children_ids.remove(child.entity_id)

    def add_parent(self, parent):
        '''Add a parent to the entity'''
        self.parent_ids.append(parent.entity_id)

    def remove_parent(self, parent: 'Entity'):
        '''Remove a parent from the entity'''
        if parent.entity_id in self.parent_ids:
            self.parent_ids.remove(parent.entity_id)


    def get_children(self) -> List['str']:
        '''Get all children of the entity'''
        return self.children_ids

    def get_parents(self) -> List['str']:
        '''Get all parents of the entity'''
        return self.parent_ids

    def remove_child_by_id(self, child_id: str):
        '''Remove a child from the entity by its ID'''
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)

    def set_attribute(self, name: str, value: Any):
        '''Set an attribute on the entity'''
        self._attributes[name] = value
    
    def set_attributes(self, attributes: Dict[str, Any]):
        '''Set multiple attributes on the entity'''
        for name, value in attributes.items():
            self.set_attribute(name, value)
    
    def get_attributes(self) -> Dict[str, Any]:
        '''Get all attributes on the entity'''
        return self._attributes

    def get_attribute(self, name: str) -> Any:
        '''Get an attribute from the entity'''
        return self._attributes[name]

    def remove_attribute(self, name: str):
        '''Remove an attribute from the entity'''
        del self._attributes[name]

    def remove_attributes(self, names: List[str]):
        '''Remove multiple attributes from the entity'''
        for name in names:
            self.remove_attribute(name)

    def has_attribute(self, name: str) -> bool:
        '''Check if an attribute exists on the entity'''
        return name in self._attributes
    
    def get_available_attributes(self) -> List[str]:
        '''Get a list of all attributes on the entity'''
        return list(self._attributes.keys())

    def get_configured_strategies(self) -> List[str]:
        '''Get a list of all configured strategies on the entity'''
        return self.configured_strategies
    
    def merge_entities(self, entities: List['Entity'], merge_config):

        entities = [self] + entities
        for config in merge_config:
            merge_method = config['method']
            attributes = config['attributes']

            if merge_method == 'concatenate':
                for attribute in attributes: 
                    self.set_attribute(attribute, np.concatenate([entity.get_attribute(attribute) for entity in entities if entity.has_attribute(attribute)]))
            if merge_method == 'take_first':
                for attribute in attributes: 
                    for entity in entities:
                        if entity.has_attribute(attribute):
                            self.set_attribute(attribute, entity.get_attribute(attribute))
                            break

    def update_strategy_requests(self, strategy_request):
        for i, request in enumerate(self.strategy_requests):
            if request.entity_id == strategy_request.entity_id:
                self.strategy_requests[i] = strategy_request
                return
        self.strategy_requests.append(strategy_request)


    def serialize(self):

        return {
            'entity_name': self.entity_name.value,
            'meta_data': {},
            'class_path': self.__class__.__module__ + '.' + self.__class__.__name__,
            'entity_id': self.entity_id,
            'entity_type': self.entity_name.value,
            'child_ids': self.get_children(),
            'strategy_requests': [request.serialize() for request in self.strategy_requests],
        }

    def to_db(self):
        """
        Converts the entity's attributes into a dictionary for database storage.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement the 'to_db' method.")

    @classmethod
    def from_db(cls, data):
        """
        Creates an entity from a database object (or dictionary).
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement the 'from_db' method.")


    @staticmethod
    def get_maximum_members():
        pass
    
    @classmethod
    def get_class_path(cls):
        return f"{cls.__module__}.{cls.__name__}"




