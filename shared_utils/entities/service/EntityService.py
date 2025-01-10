from shared_utils.cache.CacheService import CacheService
from shared_utils.entities.EnityEnum import EntityEnum


class EntityService:
    def __init__(self):
        self.cache_service = CacheService()

    def get_entity(self, entity_id):
        """Get an entity by its ID from cache or database"""
        entity = self.load_from_cache(entity_id)
        if entity is None:
            entity = self.load_from_db(entity_id)
            if entity:
                self.save_entity(entity)
        return entity

    def save_entity(self, entity):
        """Save an entity to cache"""
        self.cache_service.set(entity.entity_id, entity)

    def save_entities(self, entities):
        """Save multiple entities to cache"""
        entity_dict = {entity.entity_id: entity for entity in entities}
        self.cache_service.set_many(entity_dict)

    def load_from_cache(self, entity_id):
        """Load an entity from cache"""
        return self.cache_service.get(entity_id)

    def load_entities_from_cache(self, entity_ids):
        """Load multiple entities from cache"""
        return self.cache_service.get_many(entity_ids)

    def load_from_db(self, entity_id):
        """Load an entity from database - to be implemented by subclasses"""
        raise NotImplementedError("Load_from_db not implemented for EntityService.")

    def clear_entity(self, entity_id):
        """Remove an entity from cache"""
        self.cache_service.delete(entity_id)

    def clear_entities(self, entity_ids):
        """Remove multiple entities from cache"""
        self.cache_service.delete_many(entity_ids)
