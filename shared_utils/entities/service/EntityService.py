import importlib
import asyncio
import re

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync, sync_to_async

from shared_utils.cache.CacheService import CacheService
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.EntityModel import EntityModel
import logging
from django.db import connection
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityService:
    def __init__(self):
        self.cache_service = CacheService()
        self.channel_layer = get_channel_layer()

    def get_entity(self, entity_id):
        """Get an entity by its ID from cache or database"""
        logger.info(f"Getting entity {entity_id}")
        cached_entity = self.load_from_cache(entity_id)
        logger.info(f"Entity {entity_id} loaded from cache")

        db_entity = None
        if cached_entity is None:
            logger.info(f"Entity {entity_id} not found in cache, loading from database")
            db_entity = self.load_from_db(entity_id)

        if cached_entity is None and db_entity is None:
            logger.info(f"Entity {entity_id} not found in database")
            raise ValueError(f"Entity with ID {entity_id} not found")

        if cached_entity:
            return cached_entity
        if db_entity:
            self.cache_service.set(entity_id, db_entity)
            return db_entity

        return None

    def delete_entity(self, entity_id):
        self.clear_entity(entity_id)

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
        self._broadcast_to_global_socket({
            entity.entity_id: entity.serialize()
        })

        # Save to database
        self.save_to_db(entity)
            
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
        
    def save_to_db(self, entity):
        """Save an entity to database"""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use thread executor for sync database operations
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(self._save_to_db_sync, entity)
                        future.result()
                        return
            except RuntimeError:
                pass
            
            # We're in a sync context, still use the sync method directly
            self._save_to_db_sync(entity)
            
        except Exception as e:
            # If we get async/sync mixing errors, fall back to thread executor
            if "async event loop" in str(e) or "AsyncToSync" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._save_to_db_sync, entity)
            else:
                logger.error(f"Error saving entity {entity.entity_id} to database: {e}")
                raise

    def _save_to_db_sync(self, entity):
        """Synchronous version of save_to_db for use in thread executor"""
        try:
            model = entity.to_db()
            model.save()
            logger.info(f"Entity {entity.entity_id} saved to database")
        except Exception as e:
            logger.error(f"Error saving entity {entity.entity_id} to database: {e}")
            raise

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

    def find_entities(self, filters: list) -> list:
        """
        Find entities based on a list of filter conditions.
        
        Args:
            filters: List of filter dictionaries with keys: attribute, operator, value
            
        Returns:
            List of entity IDs matching all filter conditions
        """
        if not filters:
            return []
            
        try:
            query, params = self._build_sql_query(filters)
            
            # Execute query
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                results = [str(row[0]) for row in cursor.fetchall()]
                
            logger.info(f"Query found {len(results)} matching entities")
            return results
            
        except Exception as e:
            logger.error(f"Error executing find_entities query: {str(e)}")
            raise

    def _normalize_date_string(self, date_str: str) -> str:
        """
        Normalize date string to YYYY-MM-DD format for consistent comparison.
        Handles various input formats including YYYYMMDD, ISO with time, etc.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Normalized date string in YYYY-MM-DD format
        """
        if not date_str:
            return date_str
            
        # Remove any whitespace
        date_str = str(date_str).strip()
        
        # Handle YYYYMMDD format (8 digits)
        if re.match(r'^\d{8}$', date_str):
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
        # Handle ISO format with time (extract date part)
        if 'T' in date_str:
            date_str = date_str.split('T')[0]
            
        # Handle formats with space and time
        if ' ' in date_str:
            date_str = date_str.split(' ')[0]
            
        # Already in YYYY-MM-DD format or similar
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]  # Take only YYYY-MM-DD part
            
        # Return as-is if we can't normalize it
        return date_str

    def _build_sql_query(self, filters: list) -> tuple:
        """
        Build SQL query from filter configuration.
        
        Args:
            filters: List of filter dictionaries
            
        Returns:
            Tuple of (query_string, params_list)
        """
        base_query = "SELECT entity_id FROM shared_utils_entitymodel WHERE "
        where_clauses = []
        params = []
        
        for filter_obj in filters:
            attr = filter_obj['attribute']
            op = filter_obj['operator']
            val = filter_obj['value']
            
            # Handle special cases for direct columns (entity_type, child_ids)
            if attr in ['entity_type', 'child_ids']:
                if op == 'equals':
                    if attr == 'child_ids':
                        # For child_ids, check if the value exists in the JSONB array
                        where_clauses.append('child_ids ? %s')
                        params.append(str(val))
                    else:
                        where_clauses.append(f'{attr} = %s')
                        params.append(val)
                elif op == 'not_equals':
                    if attr == 'child_ids':
                        where_clauses.append('NOT (child_ids ? %s)')
                        params.append(str(val))
                    else:
                        where_clauses.append(f'{attr} != %s')
                        params.append(val)
                elif op == 'contains' and attr == 'entity_type':
                    # Case-insensitive contains for entity_type
                    where_clauses.append('LOWER(entity_type) LIKE LOWER(%s)')
                    params.append(f'%{val}%')
                elif op == 'in':
                    if attr == 'child_ids':
                        # For child_ids with 'in' operator, check if any of the values exist
                        conditions = []
                        for v in val:
                            conditions.append('child_ids ? %s')
                            params.append(str(v))
                        where_clauses.append(f"({' OR '.join(conditions)})")
                    else:
                        placeholders = ', '.join(['%s'] * len(val))
                        where_clauses.append(f'{attr} IN ({placeholders})')
                        params.extend(val)
            # Handle attributes stored in JSONB
            else:
                if op == 'equals':
                    where_clauses.append("attributes->>%s = %s")
                    # Handle boolean values properly - JSON booleans are lowercase when extracted as text
                    if isinstance(val, bool):
                        params.extend([attr, str(val).lower()])
                    else:
                        params.extend([attr, str(val)])
                elif op == 'not_equals':
                    where_clauses.append("attributes->>%s != %s")
                    # Handle boolean values properly - JSON booleans are lowercase when extracted as text
                    if isinstance(val, bool):
                        params.extend([attr, str(val).lower()])
                    else:
                        params.extend([attr, str(val)])
                elif op == 'contains':
                    # Case-insensitive contains for text searches
                    where_clauses.append("LOWER(attributes->>%s) LIKE LOWER(%s)")
                    params.extend([attr, f'%{val}%'])
                elif op == 'starts_with':
                    # Case-insensitive starts_with
                    where_clauses.append("LOWER(attributes->>%s) LIKE LOWER(%s)")
                    params.extend([attr, f'{val}%'])
                elif op == 'ends_with':
                    # Case-insensitive ends_with
                    where_clauses.append("LOWER(attributes->>%s) LIKE LOWER(%s)")
                    params.extend([attr, f'%{val}'])
                elif op == 'greater_than':
                    # Normalize date strings for comparison
                    normalized_val = self._normalize_date_string(val)
                    where_clauses.append(
                        "CASE "
                        "WHEN attributes->>%s ~ '^[0-9]+\\.?[0-9]*$' THEN (attributes->>%s)::numeric > %s "
                        "ELSE to_date(CASE "
                        "  WHEN attributes->>%s ~ '^[0-9]{8}$' THEN "
                        "    SUBSTRING(attributes->>%s, 1, 4) || '-' || SUBSTRING(attributes->>%s, 5, 2) || '-' || SUBSTRING(attributes->>%s, 7, 2) "
                        "  WHEN attributes->>%s ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN "
                        "    SUBSTRING(attributes->>%s, 1, 10) "
                        "  ELSE "
                        "    SPLIT_PART(SPLIT_PART(attributes->>%s, 'T', 1), ' ', 1) "
                        "  END, 'YYYY-MM-DD') > to_date(%s, 'YYYY-MM-DD') "
                        "END"
                    )
                    params.extend([attr, attr, val, attr, attr, attr, attr, attr, attr, attr, normalized_val])
                elif op == 'less_than':
                    # Normalize date strings for comparison
                    normalized_val = self._normalize_date_string(val)
                    where_clauses.append(
                        "CASE "
                        "WHEN attributes->>%s ~ '^[0-9]+\\.?[0-9]*$' THEN (attributes->>%s)::numeric < %s "
                        "ELSE to_date(CASE "
                        "  WHEN attributes->>%s ~ '^[0-9]{8}$' THEN "
                        "    SUBSTRING(attributes->>%s, 1, 4) || '-' || SUBSTRING(attributes->>%s, 5, 2) || '-' || SUBSTRING(attributes->>%s, 7, 2) "
                        "  WHEN attributes->>%s ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN "
                        "    SUBSTRING(attributes->>%s, 1, 10) "
                        "  ELSE "
                        "    SPLIT_PART(SPLIT_PART(attributes->>%s, 'T', 1), ' ', 1) "
                        "  END, 'YYYY-MM-DD') < to_date(%s, 'YYYY-MM-DD') "
                        "END"
                    )
                    params.extend([attr, attr, val, attr, attr, attr, attr, attr, attr, attr, normalized_val])
                elif op == 'between':
                    if isinstance(val, list) and len(val) == 2:
                        normalized_val0 = self._normalize_date_string(val[0])
                        normalized_val1 = self._normalize_date_string(val[1])
                        where_clauses.append(
                            "CASE "
                            "WHEN attributes->>%s ~ '^[0-9]+\\.?[0-9]*$' THEN (attributes->>%s)::numeric BETWEEN %s AND %s "
                            "ELSE to_date(CASE "
                            "  WHEN attributes->>%s ~ '^[0-9]{8}$' THEN "
                            "    SUBSTRING(attributes->>%s, 1, 4) || '-' || SUBSTRING(attributes->>%s, 5, 2) || '-' || SUBSTRING(attributes->>%s, 7, 2) "
                            "  WHEN attributes->>%s ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN "
                            "    SUBSTRING(attributes->>%s, 1, 10) "
                            "  ELSE "
                            "    SPLIT_PART(SPLIT_PART(attributes->>%s, 'T', 1), ' ', 1) "
                            "  END, 'YYYY-MM-DD') BETWEEN to_date(%s, 'YYYY-MM-DD') AND to_date(%s, 'YYYY-MM-DD') "
                            "END"
                        )
                        params.extend([attr, attr, val[0], val[1], attr, attr, attr, attr, attr, attr, attr, normalized_val0, normalized_val1])
                elif op == 'in':
                    if isinstance(val, list):
                        placeholders = ', '.join(['%s'] * len(val))
                        where_clauses.append(f"attributes->>%s IN ({placeholders})")
                        params.append(attr)
                        params.extend([str(v) for v in val])
                        
        # Join all where clauses with AND
        query = base_query + " AND ".join(where_clauses)
        
        return query, params



