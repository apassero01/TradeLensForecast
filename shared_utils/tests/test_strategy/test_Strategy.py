from unittest.mock import patch

import numpy as np
from django.test import TestCase
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy, AssignAttributesStrategy, MergeEntitiesStrategy, \
    ClusterStrategy, RetreiveSequencesStrategy, ExecuteCodeStrategy
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy.BaseStrategy import RemoveEntityStrategy
# Create test entities
class TestConcreteEntity(Entity):
    entity_name = EntityEnum.ENTITY
    
    def to_db(self):
        return {}
        
    @classmethod
    def from_db(cls, data):
        return cls()

class TestChildEntity(Entity):
    entity_name = EntityEnum.DATA_BUNDLE
    
    def to_db(self):
        return {}
        
    @classmethod
    def from_db(cls, data):
        return cls()

class CreateEntityStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        self.parent_entity = TestConcreteEntity()

    def create_strategy_request(self):
        """Helper to create a fresh strategy request for each test"""
        request = StrategyRequestEntity()
        request.strategy_name = "CreateEntityStrategy"
        request.param_config = {
            'entity_class': 'shared_utils.tests.test_strategy.test_Strategy.TestChildEntity'
        }
        return request

    def test_create_entity_new(self):
        """Test creating a new entity with generated UUID"""
        strategy_request = self.create_strategy_request()
        strategy = CreateEntityStrategy(self.executor, strategy_request)
        result = strategy.apply(self.parent_entity)
        
        # Verify entity was created
        self.assertEqual(len(self.parent_entity.children), 1)
        new_entity = self.parent_entity.children[0]
        self.assertIsInstance(new_entity, TestChildEntity)
        
        # Verify UUID was stored in param_config
        self.assertIsNotNone(result.param_config.get('entity_uuid'))
        self.assertEqual(new_entity.entity_id, result.param_config['entity_uuid'])

    def test_create_entity_recreation(self):
        """Test recreating an entity with existing UUID"""
        # First creation
        first_request = self.create_strategy_request()
        strategy = CreateEntityStrategy(self.executor, first_request)
        result = strategy.apply(self.parent_entity)
        first_uuid = result.param_config['entity_uuid']
        first_entity = self.parent_entity.children[0]
        first_path = first_entity.path
        
        # Store parent UUID
        parent_uuid = self.parent_entity.entity_id
        
        # Create new parent with same UUID
        new_parent = TestConcreteEntity(entity_id=parent_uuid)
        
        # Recreation request
        recreation_request = self.create_strategy_request()
        recreation_request.param_config['entity_uuid'] = first_uuid
        
        strategy = CreateEntityStrategy(self.executor, recreation_request)
        strategy.apply(new_parent)
        
        recreated_entity = new_parent.children[0]
        self.assertEqual(recreated_entity.entity_id, first_uuid)
        self.assertEqual(recreated_entity.path, first_path)

    def test_create_multiple_entities(self):
        """Test creating multiple entities maintains unique UUIDs"""
        strategy1 = CreateEntityStrategy(self.executor, self.create_strategy_request())
        strategy2 = CreateEntityStrategy(self.executor, self.create_strategy_request())
        
        request1 = strategy1.apply(self.parent_entity)
        request2 = strategy2.apply(self.parent_entity)
        
        uuid1 = request1.param_config['entity_uuid']
        uuid2 = request2.param_config['entity_uuid']
        
        self.assertNotEqual(uuid1, uuid2)
        self.assertEqual(len(self.parent_entity.children), 2)
        self.assertEqual(self.parent_entity.children[0].entity_id, uuid1)
        self.assertEqual(self.parent_entity.children[1].entity_id, uuid2)

class AssignAttributesStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        self.parent_entity = TestConcreteEntity()
        self.child_entity = TestChildEntity()
        self.parent_entity.add_child(self.child_entity)
        
        # Set up strategy request
        self.assign_strategy_request = StrategyRequestEntity()
        self.assign_strategy_request.strategy_name = "AssignAttributesStrategy"
        self.assign_strategy_request.param_config = {
            'child_path': self.child_entity.path,
            'attribute_map': {
                'X': [1, 2, 3],
                'y': [4, 5, 6]
            }
        }

    def test_set_attributes(self):
        """Test assigning attributes to child entity"""
        strategy = AssignAttributesStrategy(self.executor, self.assign_strategy_request)
        strategy_request = strategy.apply(self.parent_entity)
        
        # Verify strategy request is returned
        self.assertEqual(strategy_request, self.assign_strategy_request)
        
        # Verify attributes were assigned correctly
        self.assertEqual(self.child_entity.get_attribute('X'), [1, 2, 3])
        self.assertEqual(self.child_entity.get_attribute('y'), [4, 5, 6])

    def test_set_attributes_missing_child(self):
        """Test error when child entity not found"""
        self.assign_strategy_request.param_config['child_path'] = 'invalid/path'
        
        strategy = AssignAttributesStrategy(self.executor, self.assign_strategy_request)
        with self.assertRaises(ValueError) as context:
            strategy.apply(self.parent_entity)
        self.assertIn("Child entity not found at path", str(context.exception))

    def test_verify_executable(self):
        """Test verification of required config parameters"""
        strategy = AssignAttributesStrategy(self.executor, self.assign_strategy_request)
        
        # Test valid config
        self.assertTrue(strategy.verify_executable(self.parent_entity, self.assign_strategy_request))
        
        # Test missing child_path
        invalid_request = StrategyRequestEntity()
        invalid_request.param_config = {'attribute_map': {}}
        self.assertFalse(strategy.verify_executable(self.parent_entity, invalid_request))
        
        # Test missing attribute_map
        invalid_request.param_config = {'child_path': 'path'}
        self.assertFalse(strategy.verify_executable(self.parent_entity, invalid_request))


class TestRemoveEntityStrategy(TestCase):
    def setUp(self):
        self.strategy_executor = StrategyExecutor()
        self.strategy_request = StrategyRequestEntity()
        self.strategy = RemoveEntityStrategy(self.strategy_executor, self.strategy_request)
        
        # Create a parent and child entity for testing
        self.parent_entity = Entity()
        self.child_entity = Entity()
        self.parent_entity.add_child(self.child_entity)

    def test_remove_entity(self):
        # Verify child is initially in parent's children
        self.assertIn(self.child_entity, self.parent_entity.children)
        
        # Apply the remove strategy
        result = self.strategy.apply(self.child_entity)
        
        # Verify child was removed from parent
        self.assertNotIn(self.child_entity, self.parent_entity.children)
        
        # Verify strategy request is returned
        self.assertEqual(result, self.strategy_request)


class MergeEntitiesStrategyTestCase(TestCase):
    def setUp(self):
        # Create a parent entity
        self.parent = TestConcreteEntity()

        # Create children (data bundles)
        self.child1 = TestChildEntity()
        self.child2 = TestChildEntity()

        # Add children to the parent
        self.parent.add_child(self.child1)
        self.parent.add_child(self.child2)

        # Set attributes on the child entities
        # For example, arrays that we want to merge
        self.child1.set_attribute('X_train', np.array([[1, 2], [3, 4]]))
        self.child1.set_attribute('y_train', np.array([1, 0]))

        self.child2.set_attribute('X_train', np.array([[5, 6], [7, 8]]))
        self.child2.set_attribute('y_train', np.array([0, 1]))

        # Construct the strategy request
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.strategy_name = "MergeEntitiesStrategy"
        self.strategy_request.param_config = {
            'entities': [self.child1.path, self.child2.path],
            'merge_config': [
                {
                    'method': 'concatenate',
                    'attributes': ['X_train', 'y_train']
                }
            ]
        }

        self.strategy = MergeEntitiesStrategy(None, self.strategy_request)

    def test_verify_executable_missing_params(self):
        """Test that verify_executable raises ValueError if required params are missing"""
        bad_request = StrategyRequestEntity()
        bad_request.strategy_name = "MergeEntitiesStrategy"
        bad_request.param_config = {}  # Missing both 'entities' and 'merge_config'

        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.parent, bad_request)
        self.assertIn('Missing required parameter: entities', str(context.exception))

        bad_request.param_config = {'entities': []}
        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.parent, bad_request)
        self.assertIn('Missing required parameter: merge_config', str(context.exception))

    def test_apply_merge_concatenate(self):
        """Test that apply merges child attributes into the parent"""
        # Before merge, parent doesn't have X_train or y_train
        self.assertFalse(self.parent.has_attribute('X_train'))
        self.assertFalse(self.parent.has_attribute('y_train'))

        # Apply the strategy
        self.strategy.apply(self.parent)

        # After merge, parent should have concatenated arrays
        expected_X = np.array([[1, 2],
                               [3, 4],
                               [5, 6],
                               [7, 8]])
        expected_y = np.array([1, 0, 0, 1])

        np.testing.assert_array_equal(self.parent.get_attribute('X_train'), expected_X)
        np.testing.assert_array_equal(self.parent.get_attribute('y_train'), expected_y)

    def test_apply_missing_attributes(self):
        """Test behavior if one child doesn't have a certain attribute"""
        # Remove y_train from child2
        del self.child2._attributes['y_train']

        # Apply the strategy
        self.strategy.apply(self.parent)

        # Since concatenate is used, only those entities with the attribute get merged.
        # child1 has y_train, child2 doesn't. So only child1's y_train should appear.
        expected_X = np.array([[1, 2],
                               [3, 4],
                               [5, 6],
                               [7, 8]])
        expected_y = np.array([1, 0])  # Only from child1

        np.testing.assert_array_equal(self.parent.get_attribute('X_train'), expected_X)
        np.testing.assert_array_equal(self.parent.get_attribute('y_train'), expected_y)


class ClusterStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        self.entity = TestConcreteEntity()
        self.entity.set_attribute('X', np.random.rand(10, 100, 1))  # Example time-series data

    def create_strategy_request(self, cluster_type='time_series_k_means', k=3, attribute_name='X'):
        """Helper to create a fresh strategy request for each test."""
        request = StrategyRequestEntity()
        request.strategy_name = "ClusterStrategy"
        request.param_config = {
            'cluster_type': cluster_type,
            'k': k,
            'attribute_name': attribute_name
        }
        return request

    def test_clustering_success(self):
        """Test successful clustering with time_series_k_means."""
        strategy_request = self.create_strategy_request()
        strategy = ClusterStrategy(self.executor, strategy_request)

        result = strategy.apply(self.entity)

        # Verify clustering results
        cluster_centers = self.entity.get_attribute('cluster_arr')
        self.assertIsNotNone(cluster_centers)
        self.assertEqual(cluster_centers.shape[0], strategy_request.param_config['k'])  # k clusters
        self.assertEqual(cluster_centers.ndim, 3)  # Time-series clustering typical output shape

    def test_missing_attribute_name(self):
        """Test failure when 'attribute_name' is missing in the config."""
        strategy_request = self.create_strategy_request(attribute_name=None)
        del strategy_request.param_config['attribute_name']  # Remove attribute_name from config

        strategy = ClusterStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("param_config must include 'attribute_name'", str(context.exception))

    def test_unsupported_cluster_type(self):
        """Test failure with unsupported cluster type."""
        strategy_request = self.create_strategy_request(cluster_type='unsupported_cluster_type')

        strategy = ClusterStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("Unsupported cluster_type", str(context.exception))

    def test_invalid_array_attribute(self):
        """Test failure when the array is not a NumPy array."""
        self.entity.set_attribute('X', "invalid_array")  # Set invalid array type
        strategy_request = self.create_strategy_request()

        strategy = ClusterStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("The attribute must be a NumPy array", str(context.exception))

    def test_invalid_array_shape(self):
        """Test failure when the array shape is incompatible."""
        self.entity.set_attribute('X', np.random.rand(10))  # 1D array, not valid for time-series clustering
        strategy_request = self.create_strategy_request()

        strategy = ClusterStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError):
            strategy.apply(self.entity)  # tslearn should raise an error due to invalid input shape

    def test_default_request_config(self):
        """Test the default request config for ClusterStrategy."""
        config = ClusterStrategy.get_request_config()
        self.assertEqual(config['strategy_name'], 'ClusterStrategy')
        self.assertIn('cluster_type', config['param_config'])
        self.assertEqual(config['param_config']['cluster_type'], 'time_series_k_means')
        self.assertEqual(config['param_config']['k'], 3)
        self.assertEqual(config['param_config']['attribute_name'], 'X')
        


class RetrieveSequencesStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        self.entity = TestConcreteEntity()
        self.sequence_ids = [1, 2, 3]
        self.entity.set_attribute('test_row_ids', self.sequence_ids)  # Assign test sequence IDs

    def create_strategy_request(self, id_list_attribute='test_row_ids', target_attribute_name='retrieved_sequences'):
        """Helper to create a fresh strategy request for each test."""
        request = StrategyRequestEntity()
        request.strategy_name = "RetrieveSequencesStrategy"
        request.param_config = {
            "id_list_attribute": id_list_attribute,
            "feature_list": ["open", "high", "low", "close", "volume"],
            "target_attribute_name": target_attribute_name
        }
        return request

    @patch("requests.post")
    def test_retrieve_sequences_success(self, mock_post):
        """Test successful retrieval of sequences."""
        mock_response_data = [
            {"sequence_id": 1, "sliced_data": [[1, 2, 3], [4, 5, 6]]},
            {"sequence_id": 2, "sliced_data": [[7, 8, 9], [10, 11, 12]]},
            {"sequence_id": 3, "sliced_data": [[13, 14, 15], [16, 17, 18]]},
        ]
        mock_post.return_value.ok = True
        mock_post.return_value.json.return_value = mock_response_data

        strategy_request = self.create_strategy_request()
        strategy = RetreiveSequencesStrategy(self.executor, strategy_request)
        strategy.apply(self.entity)

        # Verify the retrieved sequences are stored correctly
        retrieved_sequences = self.entity.get_attribute("retrieved_sequences")
        self.assertEqual(retrieved_sequences, mock_response_data)
        mock_post.assert_called_once_with(
            'http://localhost:8000/sequenceset_manager/get_sequences_by_ids/',
            json={
                "feature_list": ["open", "high", "low", "close", "volume"],
                "sequence_ids": self.sequence_ids,
            }
        )

    def test_missing_id_list_attribute(self):
        """Test failure when 'id_list_attribute' is missing in the config."""
        strategy_request = self.create_strategy_request()
        del strategy_request.param_config['id_list_attribute']  # Remove id_list_attribute

        strategy = RetreiveSequencesStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("param_config must include 'id_list_attribute' to locate the sequence IDs", str(context.exception))

    def test_invalid_sequence_ids_type(self):
        """Test failure when the sequence IDs are not a list or tuple."""
        self.entity.set_attribute('test_row_ids', "invalid_type")  # Assign invalid type
        strategy_request = self.create_strategy_request()

        strategy = RetreiveSequencesStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("'test_row_ids' must be a list or tuple", str(context.exception))

    @patch("requests.post")
    def test_failed_api_call(self, mock_post):
        """Test failure when the API call is unsuccessful."""
        mock_post.return_value.ok = False
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        strategy_request = self.create_strategy_request()
        strategy = RetreiveSequencesStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("Request to http://localhost:8000/sequenceset_manager/get_sequences_by_ids/ failed", str(context.exception))
        mock_post.assert_called_once()

    def test_default_request_config(self):
        """Test the default request config for RetreiveSequencesStrategy."""
        config = RetreiveSequencesStrategy.get_request_config()
        self.assertEqual(config["strategy_name"], "RetreiveSequencesStrategy")
        self.assertIn("id_list_attribute", config["param_config"])
        self.assertEqual(config["param_config"]["id_list_attribute"], "test_row_ids")
        self.assertEqual(config["param_config"]["feature_list"], ["open", "high", "low", "close", "volume"])
        self.assertEqual(config["param_config"]["target_attribute_name"], "test_sequences")


class ExecuteCodeStrategyTestCase(TestCase):
    def setUp(self):
        self.executor = StrategyExecutor()
        self.entity = TestConcreteEntity()
        self.entity.set_attribute('X', np.array([1, 2, 3]))  # Assign test data
        self.entity.set_attribute('y', np.array([4, 5, 6]))  # Assign test data

    def create_strategy_request(self, code=None, result_attribute="code_result"):
        """Helper to create a fresh strategy request for each test."""
        request = StrategyRequestEntity()
        request.strategy_name = "ExecuteCodeStrategy"
        request.param_config = {
            "code": code or "code_result = entity.get_attribute('X') + entity.get_attribute('y')",
            "result_attribute": result_attribute
        }
        return request

    def test_execute_code_success(self):
        """Test successful execution of a valid code snippet."""
        strategy_request = self.create_strategy_request()
        strategy = ExecuteCodeStrategy(self.executor, strategy_request)
        strategy.apply(self.entity)

        # Verify result is stored correctly in the entity
        result = self.entity.get_attribute("code_result")
        expected_result = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected_result)

    def test_missing_code(self):
        """Test failure when no code is provided in the request."""
        strategy_request = self.create_strategy_request(code=None)
        del strategy_request.param_config['code']  # Remove the code snippet

        strategy = ExecuteCodeStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("Failed to execute code", str(context.exception))

    def test_invalid_code_syntax(self):
        """Test failure when code has invalid syntax."""
        invalid_code = "result = entity.get_attribute('X') + "  # Syntax error
        strategy_request = self.create_strategy_request(code=invalid_code)

        strategy = ExecuteCodeStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("Failed to execute code", str(context.exception))

    def test_result_not_set(self):
        """Test failure when the executed code does not set the expected result variable."""
        code_without_result = "temp = entity.get_attribute('X') + entity.get_attribute('y')"  # No 'result'
        strategy_request = self.create_strategy_request(code=code_without_result)

        strategy = ExecuteCodeStrategy(self.executor, strategy_request)

        with self.assertRaises(ValueError) as context:
            strategy.apply(self.entity)

        self.assertIn("did not set a 'result' variable", str(context.exception))

    def test_default_request_config(self):
        """Test the default request config for ExecuteCodeStrategy."""
        config = ExecuteCodeStrategy.get_request_config()
        self.assertEqual(config["strategy_name"], "ExecuteCodeStrategy")
        self.assertIn("code", config["param_config"])
        self.assertEqual(config["param_config"]["code"], "result = entity.get_attribute('X') + entity.get_attribute('y')")
        self.assertEqual(config["param_config"]["result_attribute"], "code_result")