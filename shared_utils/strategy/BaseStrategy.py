import json
from abc import ABC, abstractmethod
import importlib

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.Entity import Entity
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import requests
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
import inspect
import logging


class Strategy(ABC):
    entity_type = EntityEnum.ENTITY

    strategy_description = 'This is the base strategy class'

    logger = logging.getLogger(__name__)
    
    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor
        self.entity_service = EntityService()
        self.executor_service = StrategyExecutorService()

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
            'config': cls.get_request_config(),
            'source': inspect.getsource(cls)
        }

class GetEntityStrategy(Strategy):
    """Generic strategy for getting an entity from anywhere (cache, db, etc.)"""

    strategy_description = 'Retrieves an entity from the cache'

    def verify_executable(self, entity, strategy_request):
        pass

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        self.strategy_request.ret_val['entity'] = entity
        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
        }

class SaveEntityStrategy(Strategy):
    """Generic strategy for saving an entity to anywhere (cache, db, etc.)"""
    strategy_description = 'Saves an entity to the cache'

    def verify_executable(self, entity, strategy_request):
        pass

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        entity_service = EntityService()
        entity_service.save_entity(entity)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return

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
        self.logger.info(f"Creating entity for parent {parent_entity.entity_id}")
        config = self.strategy_request.param_config
        entity_class_path = config.get('entity_class')

        entity_uuid = config.get('entity_uuid')
        try:
            new_entity = self.entity_service.get_entity(entity_uuid)
        except ValueError:
            new_entity = None


        # Import the entity class
        if new_entity is None:
            try:
                module_path, class_name = entity_class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                entity_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import entity class: {entity_class_path}") from e

            new_entity = entity_class(entity_id=entity_uuid)
        
        # Store UUID in param_config if this is a new entity
        if entity_uuid is None:
            self.strategy_request.param_config['entity_uuid'] = new_entity.entity_id
            
        # Add as child to parent
        request_list = new_entity.on_create(self.strategy_request.param_config)
        parent_entity.add_child(new_entity)
        self.entity_service.save_entity(new_entity)
        if request_list:
            for request in request_list:
                self.entity_service.save_entity(request)
                request = self.strategy_executor.execute_request(request)
                if request.ret_val.get('child_entity'):
                    result_of_execute = request.ret_val['child_entity']
                    parent_entity.add_child(result_of_execute)

        
        self.entity_service.save_entity(new_entity)
        self.strategy_request.ret_val['child_entity'] = new_entity

        # Workaround for the fact there needs to be a request on an entity for its updated state to be returned to whose asking so need to make a mock request that this is a new entity that exists
        child_request = StrategyRequestEntity()
        child_request.target_entity_id = new_entity.entity_id
        child_request.strategy_name = "FRUAD"
        child_request.ret_val['entity'] = new_entity
        self.strategy_request.add_nested_request(child_request)
        # Add the new entity to the strategy request
        self.strategy_request.ret_val['entity'] = parent_entity
        self.logger.info(f"Entity created: {new_entity.entity_id}")
        
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
        assign_id = config.get('assign_id')
        attribute_map = config.get('attribute_map', {})
        
        # Find child entity


        target_entity = self.entity_service.get_entity(assign_id)
        if not target_entity:
            raise ValueError(f"Child entity not found at path: {assign_id}")
            
        # Assign attributes
        for source_name, target_name in attribute_map.items():
            target_entity.set_attribute(target_name, sourceEntity.get_attribute(source_name))

        self.entity_service.save_entity(target_entity)
        #TODO again sloppy but going to make a fake request so that the entity is viewed as updated on the frontend
        child_request = StrategyRequestEntity()
        child_request.target_entity_id = target_entity.entity_id
        child_request.strategy_name = "FRUAD"
        self.strategy_request.add_nested_request(child_request)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'assign_id': '',
            'attribute_map': {"source_attribute": "target_attribute"}
        }

class GetAttributesStrategy(Strategy):
    """Generic strategy for getting attributes from an entity"""

    strategy_description = 'Retrieves attributes from an entity and stores them in the request'

    def verify_executable(self, entity, strategy_request):
        return 'attribute_names' in strategy_request.param_config

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Retrieves attributes from an entity and stores them in the request

        param_config requirements:
        - attribute_names: list of attribute names to retrieve
        """
        config = self.strategy_request.param_config
        attribute_names = config.get('attribute_names', [])

        # Retrieve attributes
        for name in attribute_names:
            value = entity.get_attribute(name)
            self.strategy_request.ret_val[name] = value

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'attribute_names': []
        }

class SetAttributesStrategy(Strategy):
    """Generic strategy for setting attributes on an entity"""

    strategy_description = 'Sets attributes on an entity'


    def verify_executable(self, entity, strategy_request):
        return 'attribute_map' in strategy_request.param_config

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Sets attributes on an entity

        param_config requirements:
        - attribute_map: dict mapping attribute names to values
        """
        config = self.strategy_request.param_config
        attribute_map = config.get('attribute_map', {})

        # Set attributes
        for name, value in attribute_map.items():
            entity.set_attribute(name, value)

        self.entity_service.save_entity(entity)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'attribute_map': {"attribute_name": "attribute_value"}
        }
    
class AddChildStrategy(Strategy):
    """Generic strategy for adding a child entity to its parent"""

    strategy_description = 'Adds a child entity to its parent'

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        child_id = self.strategy_request.param_config.get('child_id')
        if child_id is None:
            raise ValueError("Missing required parameter: child_id")
        child_entity = self.entity_service.get_entity(child_id)
        entity.add_child(child_entity)
        self.entity_service.save_entity(entity)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'child_id': ''
        }

class RemoveChildStrategy(Strategy):
    """Generic strategy for removing a child entity from its parent"""

    strategy_description = 'Removes a child entity from its parent'

    def apply(self, entity: Entity) -> StrategyRequestEntity:

        child_id = self.strategy_request.param_config.get('child_id')
        if child_id is None:
            raise ValueError("Missing required parameter: child_id")
        entity.remove_child_by_id(child_id)
        child_entity = self.entity_service.get_entity(child_id)
        child_entity.remove_parent(entity)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'child_id': ''
        }

class UpdateChildrenStrategy(Strategy):
    """Generic strategy for updating all children of an entity"""

    strategy_description = 'Updates all children of an entity'

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        child_ids = self.strategy_request.param_config.get('child_ids', [])
        entity.children_ids = child_ids

        return self.strategy_request

    def verify_executable(self, entity, strategy_request):
        if 'child_ids' not in strategy_request.param_config:
            raise ValueError('Missing required parameter: child_ids')

class ExecuteRequestChildren(Strategy):
    """Generic strategy for executing another strategy request"""

    strategy_description = 'Executes another strategy request'

    def verify_executable(self, entity, strategy_request):
        return 'request' in strategy_request.param_config

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        children = entity.get_children()
        request_child_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.STRATEGY_REQUEST)

        for child in children:
            if child in request_child_ids:
                request = self.entity_service.get_entity(child)
                request = self.executor_service.execute_request(request)
                self.entity_service.save_entity(request)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        return {
            'request': None
        }




class RemoveEntityStrategy(Strategy):
    """Generic strategy for removing an entity"""

    strategy_description = 'Removes an entity from the parent entity'

    def verify_executable(self, entity, strategy_request):
        pass

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        for parent in entity.parent_ids:
            remove_child_request = StrategyRequestEntity()
            remove_child_request.target_entity_id = parent
            remove_child_request.strategy_name = RemoveChildStrategy.__name__
            remove_child_request.param_config['child_id'] = entity.entity_id
            remove_child_request = self.executor_service.execute_request(remove_child_request)
            self.strategy_request.add_nested_request(remove_child_request)

        entity.deleted = True
        self.strategy_request.ret_val['entity'] = entity


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
        target_ids = config.get('entities')
        merge_configs = config.get('merge_config')

        # strategy passes in a list of paths. We need to get all of the entities. Might be a little anti pattern
        # but for now this is what it is. We will assume that all entities are connected in the same graph via TrainingSessionEntity.
        # Therefore we can traverse backward for the passed in entity, to the TrainingSessionEntity and then get all entities from there.
        all_attributes_list = [
            attribute
            for merge_config in merge_configs
            for attribute in merge_config['attributes']
        ]
        combined_attributes = { attribute: None for attribute in all_attributes_list}

        target_ids = [entity.entity_id] + target_ids

        for id in target_ids:
            strategy_request = self.create_get_attributes(id, all_attributes_list)
            strategy_request = self.executor_service.execute_request(strategy_request)
            for config in merge_configs:
                merge_method = config['method']
                attributes = config['attributes']

                if merge_method == 'concatenate':
                    for attribute in attributes:

                            if combined_attributes[attribute] is None:
                                combined_attributes[attribute] = strategy_request.ret_val[attribute]
                            else:
                                if isinstance(combined_attributes[attribute], np.ndarray):
                                    combined_attributes[attribute] = np.concatenate((combined_attributes[attribute], strategy_request.ret_val[attribute]), axis=0)
                                if isinstance(combined_attributes[attribute], list):
                                    combined_attributes[attribute] = combined_attributes[attribute] + strategy_request.ret_val[attribute]

        for attribute, value in combined_attributes.items():
            entity.set_attribute(attribute, value)

        return self.strategy_request

    def create_get_attributes(self, entity_id, attributes):
        """
        Creates a list of GetAttributesStrategy requests for each entity in the list.
        """
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = GetAttributesStrategy.__name__
        strategy_request.target_entity_id = entity_id
        strategy_request.param_config['attribute_names'] = attributes

        return strategy_request

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



class ClusterStrategy(Strategy):
    """
    Clusters an array (of any shape) from the entity itself using the specified
    cluster algorithm. Currently supports only 'time_series_k_means'.

    After clustering, we store the cluster centers in 'cluster_arr'.
    The centers' shape will typically be (k, *rest_of_array_dims) for time_series_k_means,
    but tslearn will throw an error if the input shape is invalid for time-series clustering.
    """

    # Choose whichever entity_type is appropriate in your system
    entity_type = EntityEnum.MODEL_STAGE

    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        config = strategy_request.param_config

        # Ensure we have an attribute name
        if 'attribute_name' not in config:
            raise ValueError("param_config must include 'attribute_name' to locate the array.")

        # Ensure the entity has that attribute
        attribute_name = config['attribute_name']
        if not entity.has_attribute(attribute_name):
            raise ValueError(f"Entity does not have the attribute '{attribute_name}' for clustering.")

        # Ensure we have a cluster_type
        cluster_type = config.get('cluster_type', 'time_series_k_means')
        if cluster_type != 'time_series_k_means':
            raise ValueError(f"Unsupported cluster_type '{cluster_type}'. Only 'time_series_k_means' is supported.")

    def apply(self, entity: Entity):
        # Extract parameters from config
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config
        attribute_name = config.get('attribute_name', 'X')
        cluster_type = config.get('cluster_type', 'time_series_k_means')
        target_arr_name = config.get('target_arr_name', 'cluster_arr')
        k = config.get('k', 3)

        # Retrieve the array from the entity
        array = entity.get_attribute(attribute_name)
        if not isinstance(array, np.ndarray):
            raise ValueError("The attribute must be a NumPy array for clustering.")

        # For now, only handle time_series_k_means
        if cluster_type == 'time_series_k_means':
            # If shape is incompatible, tslearn will raise an error
            model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", random_state=42)
            model.fit(array)

            # cluster_centers_ shape depends on input; for time-series it might be (k, time_steps, features)
            cluster_centers = model.cluster_centers_

        else:
            raise ValueError(f"Unsupported cluster_type '{cluster_type}'. Only 'time_series_k_means' is supported.")

        # Store the cluster centers on the entity
        entity.set_attribute(target_arr_name, cluster_centers)

        # Return the StrategyRequestEntity to chain further strategies if needed
        return self.strategy_request

    @staticmethod
    def get_request_config() -> dict:
        """
        Default config for the ClusterStrategy.
        The user can override these keys in StrategyRequestEntity.param_config
        """
        return {
            'strategy_name': 'ClusterStrategy',
            'strategy_path': None,
            'param_config': {
                'cluster_type': 'time_series_k_means',
                'k': 3,
                'attribute_name': 'X',
                'target_arr_name': 'cluster_arr'
            }
        }
    
class RetreiveSequencesStrategy(Strategy):
    """
    Retrieves sequence data for a list of sequence IDs by making the same hard-coded
    API call used by Get_Sequence_Sets. The data is stored as-is (list order
    matching the input IDs) in an entity attribute defined by 'target_attribute_name'.
    """

    entity_type = EntityEnum.ENTITY  # Adjust if necessary

    # Hard-coded endpoint — the same one "Get_Sequence_Sets" uses
    url = 'http://localhost:8000/sequenceset_manager/get_sequences_by_ids/'

    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        """
        Validate we have the attribute containing sequence IDs,
        and it's a list or tuple.
        """
        config = strategy_request.param_config

        if "id_list_attribute" not in config:
            raise ValueError(
                "param_config must include 'id_list_attribute' to locate the sequence IDs."
            )

        id_attribute = config["id_list_attribute"]

        seq_ids = entity.get_attribute(id_attribute)
        if not isinstance(seq_ids, (list, tuple)):
            raise ValueError(
                f"'{id_attribute}' must be a list or tuple of sequence IDs. Got: {type(seq_ids)}"
            )

    def apply(self, entity: Entity):
        """
        1. Fetch sequence IDs from the entity.
        2. Make a POST request to the same endpoint that Get_Sequence_Sets uses.
        3. Store the fetched data as-is (in the same order as the IDs) under
           'target_attribute_name' in the entity.
        """
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config
        id_attribute = config.get("id_list_attribute", "test_row_ids")
        target_attribute_name = config.get("target_attribute_name", "retrieved_sequences")

        sequence_ids = entity.get_attribute(id_attribute)
        print(sequence_ids)

        # Prepare the request payload. Adjust keys to match your API exactly.
        params = {
            "feature_list": config.get("feature_list", ["open", "high", "low", "close", "volume"]),
            "sequence_ids": sequence_ids.tolist() if isinstance(sequence_ids, np.ndarray) else sequence_ids
        }

        # Send the POST request with streaming enabled
        response = requests.post(self.url, json=params, stream=True)
        if not response.ok:
            raise ValueError(
                f"Request to {self.url} failed with status {response.status_code}: {response.text}"
            )

        # Reassemble the streamed response chunks into a complete JSON string
        json_chunks = []
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            if chunk:
                json_chunks.append(chunk.decode('utf-8'))
        full_json = "".join(json_chunks)
        fetched_sequences = json.loads(full_json)

        for sequence in fetched_sequences:
            sequence_data = sequence['sliced_data']
            sequence['sliced_data'] = sequence_data

        # Store the raw list as-is, preserving order
        entity.set_attribute(target_attribute_name, fetched_sequences)

        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Default config for RetreiveSequencesStrategy.
         - 'id_list_attribute': The entity's attribute name holding the sequence IDs.
         - 'feature_list': (optional) if you want to specify which features, but here
           we are not filtering; just pass the IDs to the API call.
         - 'target_attribute_name': The attribute under which we store the returned data.
        """
        return {
            "strategy_name": "RetreiveSequencesStrategy",
            "strategy_path": None,
            "param_config": {
                "id_list_attribute": "test_row_ids",
                "feature_list": ["open", "high", "low", "close", "volume"],
                "target_attribute_name": "test_sequences"
            }
        }


class ExecuteCodeStrategy(Strategy):
    """
    Executes a code snippet provided in the 'code' parameter of the strategy request.
    The code is executed in a new local namespace, with the entity available as 'entity'.
    The result of the code execution is stored in the entity under 'result_attribute'.
    """

    entity_type = EntityEnum.ENTITY  # Adjust if necessary

    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        """
        Validate we have the code snippet to execute
        """
        pass

    def apply(self, entity):
        """
        Execute the code snippet provided in the strategy request.
        """
        code = self.strategy_request.param_config.get('code')
        result_attribute = self.strategy_request.param_config.get('result_attribute', None)

        # Prepare the local namespace with entity and necessary imports
        local_namespace = {
            'entity': entity,
            'np': __import__('numpy')  # Ensure NumPy is available in the exec context
        }

        try:
            # Execute the code in the isolated local namespace
            exec(code, {}, local_namespace)
        except Exception as e:
            raise ValueError(f"Failed to execute code: {e}")

        # Store the result in the entity
        if result_attribute:
            if result_attribute in local_namespace:
                entity.set_attribute(result_attribute, local_namespace.get(result_attribute))
            else:
                raise ValueError("The executed code did not set a 'result' variable.")

        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Default config for ExecuteCodeStrategy.
         - 'code': The code snippet to execute.
         - 'result_attribute': The attribute under which we store the result of the code execution.
        """
        return {
            "strategy_name": "ExecuteCodeStrategy",
            "strategy_path": None,
            "param_config": {
                "code": "result = np.concatenate([entity.get_attribute('predictions') ,entity.get_attribute('y_test_scaled')],axis=-1)",
                "result_attribute": "code_result"
            }
        }

class HTTPGetRequestStrategy(Strategy):
    """
    Executes an HTTP GET request to the specified URL.
    The response is stored in the entity under 'response_attribute'.
    """

    entity_type = EntityEnum.ENTITY  # Adjust if necessary

    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        """
        Validate we have the URL to fetch
        """
        pass

    def apply(self, entity):
        """
        Execute the HTTP GET request to the URL provided in the strategy request.
        """
        url = self.strategy_request.param_config.get('url')
        response_attribute = self.strategy_request.param_config.get('response_attribute', 'http_response')

        response = requests.get(url)
        if not response.ok:
            raise ValueError(
                f"Request to {url} failed "
                f"with status {response.status_code}: {response.text}"
            )

        # Store the response in the entity
        entity.set_attribute(response_attribute, response.json())
        self.strategy_request.ret_val[response_attribute] = response.json()

        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Default config for HTTPGetRequestStrategy.
         - 'url': The URL to fetch.
         - 'response_attribute': The attribute under which we store the response.
        """
        return {
            "strategy_name": "HTTPGetRequestStrategy",
            "strategy_path": None,
            "param_config": {
                "url": "https://jsonplaceholder.typicode.com/todos/1",
                "response_attribute": "http_response"
            }
        }

