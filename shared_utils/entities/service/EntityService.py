import importlib

from shared_utils.cache.CacheService import CacheService
from shared_utils.entities import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.EntityModel import EntityModel


class EntityService:
    def __init__(self):
        self.cache_service = CacheService()

    def get_entity(self, entity_id):
        """Get an entity by its ID from cache or database"""
        entity = self.load_from_cache(entity_id)
        if entity is None:
            entity = self.load_from_db(entity_id)

        if entity is None:
            raise ValueError(f"Entity with ID {entity_id} not found")

        return entity

    def save_entity(self, entity):
        """Save an entity to cache"""
        print(f"Saving entity {entity.entity_id} to cache")
        self.cache_service.set(entity.entity_id, entity)
        print(f"Entity {entity.entity_id} saved to cache")

    # def save_entities(self, entities):
    #     """Save multiple entities to cache"""
    #     entity_dict = {entity.entity_id: entity for entity in entities}
    #     self.cache_service.set_many(entity_dict)

    def set_session_id(self, session_id):
        """Set the session ID for the cache service"""
        self.cache_service.set("current_session_id", session_id)

    def get_session_id(self):
        """Get the session ID from the cache service"""
        return self.cache_service.get("current_session_id")

    def load_from_cache(self, entity_id):
        """Load an entity from cache"""
        return self.cache_service.get(entity_id)

    def load_entities_from_cache(self, entity_ids):
        """Load multiple entities from cache"""
        return self.cache_service.get_many(entity_ids)

    def get_children_ids_by_type(self, entity, entity_type: EntityEnum):
        """Get children IDs of a specific type from an entity"""
        ##TODO this kind of bad but easy
        children_ids = []
        for child_id in entity.get_children():
            child_entity = self.load_from_cache(child_id)
            if child_entity:
                if child_entity.entity_name == entity_type:
                    children_ids.append(child_id)
        return children_ids

    def load_from_db(self, entity_id):
        """Load an entity from database - to be implemented by subclasses"""
        try:
            model = EntityModel.objects.get(entity_id=entity_id)
        except EntityModel.DoesNotExist:
            return None

        return self.create_instance_from_path(model.class_path).from_db(model)


    def clear_entity(self, entity_id):
        """Remove an entity from cache"""
        self.cache_service.delete(entity_id)
        try:
            entity = EntityModel.objects.get(entity_id=entity_id)
            entity.delete()
        except EntityModel.DoesNotExist:
            pass

    def clear_entities(self, entity_ids):
        """Remove multiple entities from cache"""
        self.cache_service.delete_many(entity_ids)

    def clear_all_entities(self):
        """Remove all entities from cache"""
        self.cache_service.clear_all()

    def create_instance_from_path(self, class_path):
        """
        Dynamically imports and instantiates a class from its class path.

        :param class_path: The dot-separated path to the class (e.g., 'my_module.MyClass').
        :param args: Positional arguments to pass to the constructor.
        :param kwargs: Keyword arguments to pass to the constructor.
        :return: An instance of the class.
        """
        # Split the path into module and class
        module_name, class_name = class_path.rsplit('.', 1)

        # Import the module
        module = importlib.import_module(module_name)
        # Get the class from the module
        cls = getattr(module, class_name)
        # Call the constructor and return the instance
        return cls
