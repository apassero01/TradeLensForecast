from abc import ABC, abstractmethod
import importlib
from typing import Any, Dict

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.entities.Entity import Entity


class Strategy(ABC):
    entity_type = EntityEnum.ENTITY

    strategy_description = 'This is the base strategy class'
    
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    @abstractmethod
    def apply(self, entity):
        """Apply the strategy to the entity."""
        pass

    def verify_executable(self, entity, strategy_request):
        """Base implementation that can be overridden by child classes"""
        return True

    @staticmethod
    def get_request_config():
        return {}

    @classmethod
    def serialize(cls):
        return {
            'name': cls.__name__,
            'entity_type': cls.entity_type.value,
            'config': cls.get_request_config()
        }



class CreateEntityStrategy(Strategy):
    """Generic strategy for creating any entity type"""
    strategy_description = 'Creates a new entity and adds it as a child to the parent entity'
    
    def verify_executable(self, entity, strategy_request):
        return 'entity_class' in strategy_request.param_config

    def apply(self, parent_entity: Entity) -> StrategyRequestEntity:
        """
        Creates a new entity and adds it as a child to the parent entity
        
        param_config requirements:
        - entity_class: fully qualified path to entity class
        - entity_uuid: (optional) UUID to use for recreation
        """
        config = self.strategy_request.param_config
        entity_class_path = config.get('entity_class')
        
        # Import the entity class
        try:
            module_path, class_name = entity_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            entity_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import entity class: {entity_class_path}") from e
        
        # Create entity with UUID if provided (recreation case)
        entity_uuid = config.get('entity_uuid')
        new_entity = entity_class(entity_id=entity_uuid)
        
        # Store UUID in param_config if this is a new entity
        if entity_uuid is None:
            self.strategy_request.param_config['entity_uuid'] = new_entity.entity_id
            
        # Add as child to parent
        new_entity.on_create(self.strategy_request.param_config)
        
        parent_entity.add_child(new_entity)

        # Add the new entity to the strategy request
        self.strategy_request.ret_val['entity'] = new_entity
        
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'entity_class': '',
            'entity_uuid': None  # Added to show it's an expected config option
        }


class AssignAttributesStrategy(Strategy):
    """Generic strategy for assigning attributes between entities"""

    strategy_description = 'Assigns attributes from parent to child entity based on mapping'
        
    def verify_executable(self, entity, strategy_request):
        config = strategy_request.param_config
        return all(key in config for key in ['child_path', 'attribute_map'])

    def apply(self, sourceEntity: Entity) -> Entity:
        """
        Assigns attributes from parent to child entity based on mapping
        
        param_config requirements:
        - child_path: path to child entity
        - attribute_map: dict mapping child attribute names to values
        """
        config = self.strategy_request.param_config
        child_path = config.get('target_path')
        attribute_map = config.get('attribute_map', {})
        
        # Find child entity
        parentEntity = sourceEntity.get_parent()
        while parentEntity.get_parent() is not None:
            parentEntity = parentEntity.get_parent()


        target_entity = parentEntity.find_entity_by_path(child_path)
        if not target_entity:
            raise ValueError(f"Child entity not found at path: {child_path}")
            
        # Assign attributes
        for source_name, target_name in attribute_map.items():
            target_entity.set_attribute(target_name, sourceEntity.get_attribute(source_name))
                
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'target_path': '',
            'attribute_map': {"source_attribute": "target_attribute"}
        }


class RemoveEntityStrategy(Strategy):
    """Generic strategy for removing an entity"""

    strategy_description = 'Removes an entity from the parent entity'

    def verify_executable(self, entity, strategy_request):
        pass

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        parent_entity = entity._parent
        parent_entity.remove_child(entity)
        return self.strategy_request
    
    @staticmethod
    def get_request_config():
        return {}


class MergeEntitiesStrategy(Strategy):
    """Generic strategy for merging multiple entities"""

    strategy_description = 'Merges multiple entities into the single parent entity'

    def verify_executable(self, entity, strategy_request):
        if 'entities' not in strategy_request.param_config:
            raise ValueError('Missing required parameter: entities')

        if 'merge_config' not in strategy_request.param_config:
            raise ValueError('Missing required parameter: merge_config')

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        config = self.strategy_request.param_config
        entity_path_list = config.get('entities')
        merge_config = config.get('merge_config')

        # strategy passes in a list of paths. We need to get all of the entities. Might be a little anti pattern
        # but for now this is what it is. We will assume that all entities are connected in the same graph via TrainingSessionEntity.
        # Therefore we can traverse backward for the passed in entity, to the TrainingSessionEntity and then get all entities from there.
        parent_entity = entity
        while parent_entity._parent is not None:
            parent_entity = parent_entity._parent

        entities = parent_entity.find_entities_by_paths(entity_path_list)
        entities = [entities[path] for path in entity_path_list]
        # Merge entities
        entity.merge_entities(entities, merge_config)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'entities': [],
            'merge_config': [
                {
                    'method': 'concatenate',
                    'attributes': ['X_train', 'y_train', 'X_test', 'y_test', 'X', 'y']
                }
            ]
        }

