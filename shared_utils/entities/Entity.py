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
        self.children: List[Entity] = []
        self._parent: Optional[Entity] = None
        self._path: Optional[str] = None
        self._attributes: Dict[str, Any] = {}

    
    def on_create(self, param_config: Optional[Dict[str, Any]] = None):
        pass

    @property
    def path(self):
        '''Get the full path to the entity'''
        if self._path is None:
            if self._parent is None:
                self._path = self.entity_name.value+":"+self.entity_id
            else:
                self._path = f"{self._parent.path}/{self.entity_name.value}:{self.entity_id}"
        return self._path

    def add_child(self, child: 'Entity'):
        '''Add a child to the entity'''
        # If child already has a parent, remove it from that parent's children
        if child._parent is not None:
            child._parent.children.remove(child)
        
        # Add to new parent
        self.children.append(child)
        child._parent = self
        
        # Invalidate path cache since the parent relationship changed
        child._path = None
    
    def remove_child(self, child: 'Entity'):
        '''Remove a child from the entity'''
        self.children.remove(child)
        child._parent = None
        child._path = None
        # delete child from memory
        del child
    
    def find_entities_by_paths(self, paths: List[str]) -> Dict[str, Optional['Entity']]:
        """
        Find multiple entities by their paths in a single traversal
        Returns a dictionary mapping paths to found entities (None if not found)
        """
        results = {path: None for path in paths}
        paths_set = set(paths)  # Convert to set for O(1) lookups
        
        # Check if current entity matches any paths
        if self.path in paths_set:
            results[self.path] = self

        # If we've found all paths, return early
        if all(results.values()):
            return results

        # Recursively search children
        for child in self.children:
            # Skip this branch if child's path isn't a prefix of any remaining unfound paths
            unfound_paths = [p for p in paths if results[p] is None]
            if not any(p.startswith(child.path) for p in unfound_paths):
                continue
                
            child_results = child.find_entities_by_paths(unfound_paths)
            # Update results with found entities
            for path, entity in child_results.items():
                if entity is not None:
                    results[path] = entity

        return results

    def find_entity_by_path(self, path: str) -> Optional['Entity']:
        """Find a single entity by path (uses find_entities_by_paths internally)"""
        results = self.find_entities_by_paths([path])
        return results[path]
    
    def get_children_by_type(self, entity_type: EntityEnum) -> List['Entity']:
        '''Get all children of a given type'''
        return [child for child in self.children if child.entity_name == entity_type]

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

    def has_attribute(self, name: str) -> bool:
        '''Check if an attribute exists on the entity'''
        return name in self._attributes
    
    def get_available_attributes(self) -> List[str]:
        '''Get a list of all attributes on the entity'''
        return list(self._attributes.keys())

    def get_parent(self) -> Optional['Entity']:
        '''Get the parent of the entity'''
        return self._parent
    
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


    def serialize(self):
        serialized_children = []
        for child in self.children:
            serialized_children.append(child.serialize())
        return {
            'entity_name': self.entity_name.value,
            'children': serialized_children,
            'meta_data': {},
            'path': self.path,
            'class_path': self.__class__.__module__ + '.' + self.__class__.__name__
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




