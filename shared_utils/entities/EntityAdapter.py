from typing import Optional, Type
from django.db import models
from django.contrib.postgres.fields import JSONField
from shared_utils.entities.Entity import Entity

class EntityAdapter:
    """
    Default adapter implementation for converting between Entity objects and Django models.
    Can be used directly or inherited for custom behavior.
    """
    
    @classmethod
    def model_to_entity(cls, model: models.Model, entity_class: Type[Entity]) -> Entity:
        """Convert a Django model instance to an Entity"""
        entity = entity_class(entity_id=str(model.id))
        
        # Set attributes from JSON field
        if hasattr(model, '_attributes'):
            entity.set_attributes(model._attributes)
            
        # Set relationships
        if hasattr(model, 'children_ids'):
            entity.children_ids = model.children_ids
            
        if hasattr(model, 'parent_ids'):
            entity.parent_ids = model.parent_ids
            
        if hasattr(model, 'configured_strategies'):
            entity.configured_strategies = model.configured_strategies
            
        return entity

    @classmethod
    def entity_to_model(cls, entity: Entity, model: Optional[models.Model] = None, 
                       model_class: Optional[Type[models.Model]] = None) -> models.Model:
        """Convert an Entity to a Django model instance"""
        if model is None:
            if model_class is None:
                raise ValueError("Either model or model_class must be provided")
                
            if entity.entity_id:
                try:
                    model = model_class.objects.get(id=entity.entity_id)
                except model_class.DoesNotExist:
                    model = model_class()
            else:
                model = model_class()

        # Update core fields
        if hasattr(model, '_attributes'):
            model._attributes = entity.get_attributes()
            
        if hasattr(model, 'entity_type'):
            model.entity_type = entity.entity_name.value
            
        if hasattr(model, 'children_ids'):
            model.children_ids = entity.get_children()
            
        if hasattr(model, 'parent_ids'):
            model.parent_ids = entity.get_parents()
            
        if hasattr(model, 'configured_strategies'):
            model.configured_strategies = entity.get_configured_strategies()
            
        if hasattr(model, 'class_path'):
            model.class_path = entity.get_class_path()

        return model 