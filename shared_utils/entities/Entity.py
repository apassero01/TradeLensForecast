from abc import ABC, abstractmethod

from shared_utils.entities.EnityEnum import EntityEnum


class Entity(ABC):
    entity_name = EntityEnum.ENTITY

    def __init__(self):
        # Initialize entity_map as an instance variable
        self.entity_map = {}

    def get_entity_map(self):
        """Return the underlying data."""
        return self.entity_map

    def set_entity_map(self, entity_map):
        """Add key value pairs to the entity map."""
        if not isinstance(entity_map, dict):
            raise ValueError("entity_map must be a dictionary")
        for key, value in entity_map.items():
            self.entity_map[key] = value

    def get_entity(self, key):
        """Return the underlying data."""
        if key not in self.entity_map:
            raise ValueError(f"Key {key} not found in entity map")
        return self.entity_map[key]

    @staticmethod
    def get_maximum_members():
        pass




