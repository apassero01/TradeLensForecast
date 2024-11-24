from abc import ABC, abstractmethod

from shared_utils.entities.EnityEnum import EntityEnum


class Entity(ABC):
    entity_name = EntityEnum.ENTITY
    registered_strategies = []

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

    def serialize(self):
        serialized_children = []
        for key, value in self.entity_map.items():
                if isinstance(value, Entity):
                    serialized_children.append(value.serialize())
                else:
                    for v in value:
                        serialized_children.append(v.serialize())
        return {
            'entity_name': self.entity_name.value,
            'children': serialized_children,
            'meta_data': {}
        }

    @classmethod
    def registered_strategies(cls):
        return [cls.registered_strategies.__name__ for strategy in cls.registered_strategies]

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




