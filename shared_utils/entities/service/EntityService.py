from shared_utils.cache.CacheService import CacheService
from shared_utils.entities.EnityEnum import EntityEnum


class EntityService:
    def __init__(self):
        self.cache_service = CacheService()

    def get_entity(self, entity_id):
        """Get an entity by its ID from cache or database"""
        print(f"Getting entity {entity_id} from cache")
        entity = self.load_from_cache(entity_id)
        print(f"Entity {entity_id} found in cache: {entity.entity_name if entity else None}")

        return entity

    def save_entity(self, entity):
        """Save an entity to cache"""
        print(f"Saving entity {entity.entity_id} to cache")
        self.cache_service.set(entity.entity_id, entity)
        print(f"Entity {entity.entity_id} saved to cache")

    def save_entities(self, entities):
        """Save multiple entities to cache"""
        entity_dict = {entity.entity_id: entity for entity in entities}
        self.cache_service.set_many(entity_dict)

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
        raise NotImplementedError("Load_from_db not implemented for EntityService.")

    def clear_entity(self, entity_id):
        """Remove an entity from cache"""
        self.cache_service.delete(entity_id)

    def clear_entities(self, entity_ids):
        """Remove multiple entities from cache"""
        self.cache_service.delete_many(entity_ids)

    def clear_all_entities(self):
        """Remove all entities from cache"""
        self.cache_service.clear_all()