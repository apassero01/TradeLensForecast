import importlib
import asyncio
from imghdr import tests

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync, sync_to_async

from shared_utils.cache.CacheService import CacheService
from shared_utils.entities import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.EntityModel import EntityModel
import logging

logger = logging.getLogger(__name__)

class EntityService:
    def __init__(self):
        self.cache_service = CacheService()
        self.channel_layer = get_channel_layer()

    def get_entity(self, entity_id):
        """Get an entity by its ID from cache or database"""
        logger.info(f"Getting entity {entity_id}")
        entity = self.load_from_cache(entity_id)
        logger.info(f"Entity {entity_id} loaded from cache")
        if entity is None:
            logger.info(f"Entity {entity_id} not found in cache, loading from database")
            entity = self.load_from_db(entity_id)

        if entity is None:
            logger.info(f"Entity {entity_id} not found in database")
            raise ValueError(f"Entity with ID {entity_id} not found")

        return entity

    def save_entity(self, entity):
        """Save an entity to cache and broadcast update via WebSocket"""
        # Save entity to cache
        logger.info(f"Saving entity {entity.entity_id} to cache")
        self.cache_service.set(entity.entity_id, entity)
        logger.info(f"Entity {entity.entity_id} saved to cache")
        
        # Check if entity socket exists
        socket_exists = self._check_entity_socket_exists(entity.entity_id)

        if hasattr(entity, 'deleted') and entity.deleted:
            self.clear_entity(entity.entity_id)
            return
        # Broadcast update
        # if not socket_exists:
        # No socket exists, broadcast to global to establish connection
        # print(f"No socket exists for entity {entity.entity_id}, broadcasting to global")
        self._broadcast_to_global_socket({
            entity.entity_id: entity.serialize()
        })
        # else:
            # Socket exists, send update through entity-specific socket
        print(f"Socket exists for entity {entity.entity_id}, broadcasting to entity socket")
        self._broadcast_to_entity_socket(entity)
            
        print(f"Entity {entity.entity_id} saved and broadcast")

    def _check_entity_socket_exists(self, entity_id):
        """Check if an entity-specific socket group exists"""
        try:
            # Get the group name for this entity
            group_name = f"entity_{entity_id}"
            
            # Use the channel layer's internal group storage to check if the group exists
            # and has any members
            if hasattr(self.channel_layer, 'groups'):
                # For InMemoryChannelLayer
                return bool(self.channel_layer.groups.get(group_name))
            elif hasattr(self.channel_layer, '_groups'):
                # For RedisChannelLayer
                return bool(async_to_sync(self.channel_layer._groups.group_channels)(group_name))
            
            # Default to False if we can't determine
            return False
            
        except Exception as e:
            print(f"Error checking socket existence: {str(e)}")
            return False

    def clear_entity(self, entity_id):
        """Remove an entity from cache and broadcast deletion"""
        print(f"Clearing entity {entity_id} from cache")
        self.cache_service.delete(entity_id)
        
        # Always broadcast deletion to both sockets to ensure cleanup
        deletion_message = {
            entity_id: {
                'deleted': True,
                'id': entity_id
            }
        }
        
        # Check if entity socket exists before broadcasting
        if self._check_entity_socket_exists(entity_id):
            self._broadcast_to_entity_socket_by_id(entity_id, deletion_message)
        
        # Always broadcast to global to ensure all clients know about deletion
        self._broadcast_to_global_socket(deletion_message)
        
        try:
            print(f"Deleting entity {entity_id} from database")
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use thread executor for sync database operations
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._delete_entity_from_db_sync, entity_id)
                        future.result()
                        return
            except RuntimeError:
                pass
            
            # We're in a sync context, use the original approach
            get_entity_model = sync_to_async(EntityModel.objects.get)
            delete_entity = sync_to_async(lambda entity: entity.delete())
            entity = async_to_sync(get_entity_model)(entity_id=entity_id)
            async_to_sync(delete_entity)(entity)
        except EntityModel.DoesNotExist:
            pass
        except Exception as e:
            # If we get async/sync mixing errors, fall back to thread executor
            if "async event loop" in str(e) or "AsyncToSync" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._delete_entity_from_db_sync, entity_id)
                    future.result()
            else:
                print(f"Error deleting entity {entity_id} from database: {str(e)}")

    def _delete_entity_from_db_sync(self, entity_id):
        """Synchronous version of deleting entity from database for use in thread executor"""
        try:
            entity = EntityModel.objects.get(entity_id=entity_id)
            entity.delete()
        except EntityModel.DoesNotExist:
            pass

    def _broadcast_to_global_socket(self, entities_data):
        """Broadcast to global WebSocket"""
        try:
            async_to_sync(self.channel_layer.group_send)(
                "global_entities",
                {
                    "type": "entity_update",
                    "entities": entities_data
                }
            )
        except Exception as e:
            print(f"Error broadcasting to global socket: {str(e)}")

    def _broadcast_to_entity_socket(self, entity):
        """Broadcast to entity-specific WebSocket"""
        try:
            async_to_sync(self.channel_layer.group_send)(
                f"entity_{entity.entity_id}",
                {
                    "type": "entity_update",
                    "entity": entity.serialize()
                }
            )
        except Exception as e:
            print(f"Error broadcasting to entity socket: {str(e)}")

    def _broadcast_to_entity_socket_by_id(self, entity_id, data):
        """Broadcast to entity-specific WebSocket using just the ID"""
        print(f"Broadcasting to entity socket {entity_id}")
        try:
            async_to_sync(self.channel_layer.group_send)(
                f"entity_{entity_id}",
                {
                    "type": "entity_update",
                    "entity": data
                }
            )
        except Exception as e:
            print(f"Error broadcasting to entity socket: {str(e)}")

    def entity_exists_in_db(self, entity_id):
        """Check if entity exists in database"""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use thread executor for sync database operations
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._entity_exists_in_db_sync, entity_id)
                        return future.result()
            except RuntimeError:
                pass
            
            # We're in a sync context, use the original approach
            get_entity_model = sync_to_async(EntityModel.objects.get)
            async_to_sync(get_entity_model)(entity_id=entity_id)
            return True
        except EntityModel.DoesNotExist:
            return False
        except Exception as e:
            # If we get async/sync mixing errors, fall back to thread executor
            if "async event loop" in str(e) or "AsyncToSync" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._entity_exists_in_db_sync, entity_id)
                    return future.result()
            raise

    def _entity_exists_in_db_sync(self, entity_id):
        """Synchronous version of entity_exists_in_db for use in thread executor"""
        try:
            EntityModel.objects.get(entity_id=entity_id)
            return True
        except EntityModel.DoesNotExist:
            return False

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
                try:
                    child_entity = self.get_entity(child_id)
                    if child_entity:
                        if child_entity.entity_name == entity_type:
                            children_ids.append(child_id)
                except ValueError:
                    # Entity not found in cache or database
                    logger.error(f"Entity with ID {child_id} not found")
                    continue

        return children_ids

    def load_from_db(self, entity_id):
        """Load an entity from database"""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use thread executor for sync database operations
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._load_from_db_sync, entity_id)
                        return future.result()
            except RuntimeError:
                pass
            
            # We're in a sync context, use the original approach
            get_entity_model = sync_to_async(EntityModel.objects.get)
            model = async_to_sync(get_entity_model)(entity_id=entity_id)
            
        except EntityModel.DoesNotExist:
            logger.error(f"Entity with ID {entity_id} not found in database")
            return None
        except Exception as e:
            # If we get async/sync mixing errors, fall back to thread executor
            if "async event loop" in str(e) or "AsyncToSync" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._load_from_db_sync, entity_id)
                    return future.result()
            raise

        return self.create_instance_from_path(model.class_path).from_db(model)

    def _load_from_db_sync(self, entity_id):
        """Synchronous version of load_from_db for use in thread executor"""
        try:
            model = EntityModel.objects.get(entity_id=entity_id)
            return self.create_instance_from_path(model.class_path).from_db(model)
        except EntityModel.DoesNotExist:
            logger.error(f"Entity with ID {entity_id} not found in database")
            return None

    def clear_entities(self, entity_ids):
        """Remove multiple entities from cache"""
        self.cache_service.delete_many(entity_ids)

    def clear_all_entities(self):
        """Remove all entities from cache"""
        self.cache_service.clear_all()

    def delete_session(self):
        """Delete the current session"""
        session_id = self.get_session_id()
        if session_id:
            self.clear_all_entities()
            self.cache_service.delete("current_session_id")
            self.delete_session_db(session_id)
        else:
            raise ValueError("No session ID set")

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

    def delete_session_db(self, session_id):
        """Delete the current session from the database"""
        all_entities = self.recurse_children(session_id)
        
        # Check if we're in an async context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use thread executor for sync database operations
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._delete_session_entities_sync, all_entities)
                    future.result()
                    self.clear_all_entities()
                    return
        except RuntimeError:
            pass
        
        # We're in a sync context, use the original approach
        for entity in all_entities:
            try:
                # Wrap the synchronous database operations with sync_to_async
                get_entity_model = sync_to_async(EntityModel.objects.get)
                delete_entity = sync_to_async(lambda entity_model: entity_model.delete())
                entity_model = async_to_sync(get_entity_model)(entity_id=entity.entity_id)
                async_to_sync(delete_entity)(entity_model)
            except EntityModel.DoesNotExist:
                pass
            except Exception as e:
                # If we get async/sync mixing errors, fall back to thread executor
                if "async event loop" in str(e) or "AsyncToSync" in str(e):
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._delete_session_entities_sync, all_entities)
                        future.result()
                        break
                else:
                    print(f"Error deleting entity {entity.entity_id} from database: {str(e)}")

        self.clear_all_entities()

    def _delete_session_entities_sync(self, all_entities):
        """Synchronous version of deleting session entities for use in thread executor"""
        for entity in all_entities:
            try:
                entity_model = EntityModel.objects.get(entity_id=entity.entity_id)
                entity_model.delete()
            except EntityModel.DoesNotExist:
                pass

    def recurse_children(self, entity_id, entity_type = None):
        """Recursively get all children of a specific type until no children are left."""
        entity = self.load_from_db(entity_id)

        if not entity:
            return []

        entities = []

        # Add the current entity if it matches the type
        if entity_type:
            if entity.entity_name == entity_type:
                entities.append(entity)
        else:
            entities.append(entity)

        # If there are no children, return the current entities
        if not entity.children_ids:
            return entities

        # Recursively get children entities
        for child_id in entity.children_ids:
            child_entities = self.recurse_children(child_id, entity_type)
            entities.extend(child_entities)

        return entities



