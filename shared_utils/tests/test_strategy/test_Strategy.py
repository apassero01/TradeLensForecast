import numpy as np
from django.test import TestCase
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy, AssignAttributesStrategy, MergeEntitiesStrategy
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

    def test_assign_attributes(self):
        """Test assigning attributes to child entity"""
        strategy = AssignAttributesStrategy(self.executor, self.assign_strategy_request)
        strategy_request = strategy.apply(self.parent_entity)
        
        # Verify strategy request is returned
        self.assertEqual(strategy_request, self.assign_strategy_request)
        
        # Verify attributes were assigned correctly
        self.assertEqual(self.child_entity.get_attribute('X'), [1, 2, 3])
        self.assertEqual(self.child_entity.get_attribute('y'), [4, 5, 6])

    def test_assign_attributes_missing_child(self):
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