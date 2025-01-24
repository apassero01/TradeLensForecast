from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type
from shared_utils.entities.EnityEnum import EntityEnum
from uuid import uuid4, UUID
import numpy as np
from django.db import models
from shared_utils.entities.EntityModel import EntityModel
class Entity():
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

    def to_db(self, model=None):
        """Convert entity to database model"""
        return EntityAdapter.entity_to_model(self, model)
    
    @classmethod
    def from_db(cls, model):
        """Create entity from database model"""
        return EntityAdapter.model_to_entity(model, cls)

    @staticmethod
    def get_maximum_members():
        pass
    
    @classmethod
    def get_class_path(cls):
        return f"{cls.__module__}.{cls.__name__}"


class EntityAdapter:
    """
    Default adapter implementation for converting between Entity objects and Django models.
    Can be used directly or inherited for custom behavior.
    """

    @classmethod
    def model_to_entity(cls, model: models.Model, entity_class: Type['Entity'] = Entity) -> 'Entity':
        """Convert a Django model instance to an Entity"""
        entity = entity_class(entity_id=str(model.entity_id))

        # Set attributes from JSON field
        if hasattr(model, 'attributes'):
            entity.set_attributes(model.attributes)

        # Set relationships
        if hasattr(model, 'children_ids'):
            entity.children_ids = model.children_ids

        if hasattr(model, 'parent_ids'):
            entity.parent_ids = model.parent_ids

        entity.strategy_requests = cls.load_strategy_requests(model)

        return entity

    @classmethod
    def entity_to_model(cls, entity: 'Entity', model: Optional[models.Model] = None,
                       model_class: Type[models.Model] = EntityModel) -> models.Model:
        """Convert an Entity to a Django model instance"""
        if model is None:
            if entity.entity_id:
                try:
                    model = model_class.objects.get(entity_id=entity.entity_id)
                except model_class.DoesNotExist:
                    model = model_class(entity_id=entity.entity_id)
            else:
                model = model_class(entity.entity_id)

        # Update core fields
        if hasattr(model, 'attributes'):
            model.attributes = entity.get_attributes()

        if hasattr(model, 'entity_type'):
            model.entity_type = entity.entity_name.value

        if hasattr(model, 'children_ids'):
            model.children_ids = entity.get_children()

        if hasattr(model, 'parent_ids'):
            model.parent_ids = entity.get_parents()

        if hasattr(model, 'class_path'):
            model.class_path = entity.get_class_path()

        cls.save_strategy_requests(entity, model)

        return model
    
    @classmethod
    def load_strategy_requests(cls, model: models.Model):
        from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
        from shared_utils.models import StrategyRequest
        strategy_requests = StrategyRequest.objects.filter(entity_model=model)
        return [StrategyRequestEntity.from_db(strategy_request) for strategy_request in strategy_requests]

    @classmethod
    def save_strategy_requests(cls, entity, model: models.Model):
        for strategy_entity in entity.strategy_requests:
            strategy_model = strategy_entity.to_db()
            strategy_model.entity_model = model
            strategy_model.save()




