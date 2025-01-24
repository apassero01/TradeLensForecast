from django.test import TestCase
from shared_utils.entities.Entity import Entity, EntityAdapter
from shared_utils.entities.EntityModel import EntityModel
from shared_utils.entities.EnityEnum import EntityEnum
import numpy as np
import uuid
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.models import StrategyRequest
from django.utils import timezone

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

class EntityTestCase(TestCase):
    def setUp(self):
        # Create a hierarchy of test entities
        self.parent = TestConcreteEntity()
        self.child1 = TestChildEntity()
        self.child2 = TestChildEntity()
        
        # Set up test data
        self.child1.set_attribute('X_train', np.array([[1, 2], [3, 4]]))
        self.child1.set_attribute('y_train', np.array([1, 0]))

        self.child2.set_attribute('X_train', np.array([[5, 6], [7, 8]]))
        self.child2.set_attribute('y_train', np.array([0, 1]))

    def test_set_and_get_attribute(self):
        """Test setting and getting attributes"""
        entity = TestConcreteEntity()
        
        # Test basic attribute setting and getting
        entity.set_attribute('test_key', 'test_value')
        self.assertEqual(entity.get_attribute('test_key'), 'test_value')
        
        # Test with different types of values
        test_cases = {
            'int_value': 42,
            'float_value': 3.14,
            'list_value': [1, 2, 3],
            'dict_value': {'key': 'value'},
            'bool_value': True,
            'none_value': None
        }
        
        for key, value in test_cases.items():
            entity.set_attribute(key, value)
            self.assertEqual(entity.get_attribute(key), value)

    def test_get_attribute_keyerror(self):
        """Test getting non-existent attribute raises KeyError"""
        entity = TestConcreteEntity()
        
        with self.assertRaises(KeyError):
            entity.get_attribute('nonexistent_key')

    def test_has_attribute(self):
        """Test checking if attributes exist"""
        entity = TestConcreteEntity()
        
        # Test with non-existent attribute
        self.assertFalse(entity.has_attribute('test_key'))
        
        # Test after setting attribute
        entity.set_attribute('test_key', 'test_value')
        self.assertTrue(entity.has_attribute('test_key'))

    def test_get_available_attributes(self):
        """Test getting list of available attributes"""
        entity = TestConcreteEntity()
        
        # Test with no attributes
        self.assertEqual(entity.get_available_attributes(), [])
        
        # Test with multiple attributes
        entity.set_attribute('key1', 'value1')
        entity.set_attribute('key2', 'value2')
        
        available_attrs = entity.get_available_attributes()
        self.assertEqual(len(available_attrs), 2)
        self.assertIn('key1', available_attrs)
        self.assertIn('key2', available_attrs)

    def test_parent_child_relationship(self):
        """Test parent-child relationship is correctly established"""
        self.parent.add_child(self.child1)
        
        self.assertEqual(self.child1.parent_ids[0], self.parent.entity_id)
        self.assertIn(self.child1.entity_id, self.parent.children_ids)

    def test_merge_entities_concatenate(self):
        """Test merging entities using the concatenate method"""
        # The parent does not have X_train or y_train set at the start
        # We'll merge child1 and child2 into parent
        merge_config = [
            {
                'method': 'concatenate',
                'attributes': ['X_train', 'y_train']
            }
        ]

        self.parent.merge_entities([self.child1, self.child2], merge_config)

        # X_train should be vertically concatenated: shape (4,2)
        expected_X = np.array([[1, 2],
                               [3, 4],
                               [5, 6],
                               [7, 8]])
        # y_train should be concatenated: shape (4,)
        expected_y = np.array([1, 0, 0, 1])

        np.testing.assert_array_equal(self.parent.get_attribute('X_train'), expected_X)
        np.testing.assert_array_equal(self.parent.get_attribute('y_train'), expected_y)

    def test_merge_entities_take_first(self):
        """Test merging entities using the take_first method"""
        # Let's say only child1 has an attribute 'metadata', child2 does not
        self.child1.set_attribute('metadata', {'info': 'child1_info'})

        merge_config = [
            {
                'method': 'take_first',
                'attributes': ['metadata', 'X_train']
            }
        ]

        # Merge them, parent + [child1, child2]
        # For 'metadata', parent doesn't have it, child1 does, so take from child1.
        # For 'X_train', parent doesn't have it, child1 does, so take from child1 (ignore child2).
        
        self.parent.merge_entities([self.child1, self.child2], merge_config)

        self.assertEqual(self.parent.get_attribute('metadata'), {'info': 'child1_info'})
        np.testing.assert_array_equal(self.parent.get_attribute('X_train'), np.array([[1, 2], [3, 4]]))

    def test_remove_attribute(self):
        """Test removing a single attribute from the entity"""
        entity = TestConcreteEntity()
        entity.set_attribute('test_key', 'test_value')
        self.assertTrue(entity.has_attribute('test_key'))  # Verify attribute exists

        entity.remove_attribute('test_key')  # Remove the attribute
        self.assertFalse(entity.has_attribute('test_key'))  # Verify attribute is removed

    def test_remove_attributes(self):
        """Test removing multiple attributes from the entity"""
        entity = TestConcreteEntity()
        entity.set_attribute('key1', 'value1')
        entity.set_attribute('key2', 'value2')
        entity.set_attribute('key3', 'value3')

        self.assertTrue(entity.has_attribute('key1'))
        self.assertTrue(entity.has_attribute('key2'))
        self.assertTrue(entity.has_attribute('key3'))

        # Remove multiple attributes
        entity.remove_attributes(['key1', 'key2'])

        # Verify the attributes are removed
        self.assertFalse(entity.has_attribute('key1'))
        self.assertFalse(entity.has_attribute('key2'))
        self.assertTrue(entity.has_attribute('key3'))  # key3 should still exist

class TestEntityAdapter(TestCase):
    def setUp(self):
        self.test_uuid = str(uuid.uuid4())
        self.test_attributes = {
            'name': 'Test Entity',
            'value': 42
        }
        self.test_children_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.test_parent_ids = [str(uuid.uuid4())]
        
        # Create a test model instance
        self.test_model = EntityModel(
            entity_id=self.test_uuid,
            entity_type='ENTITY',
            attributes=self.test_attributes,
            children_ids=self.test_children_ids,
            parent_ids=self.test_parent_ids,
            class_path='shared_utils.entities.Entity.Entity'
        )

        # Add setup for strategy request testing
        self.strategy_request = StrategyRequest.objects.create(
            strategy_name="test_strategy",
            param_config={"param1": "value1"},
            entity_model=self.test_model,
            target_entity_id=str(uuid.uuid4()),
            add_to_history=True,
            created_at=timezone.now(),
            updated_at=timezone.now()
        )

        # Create a nested strategy request
        self.nested_request = StrategyRequest.objects.create(
            strategy_name="nested_strategy",
            param_config={"param2": "value2"},
            parent_request=self.strategy_request,
            add_to_history=False
        )

    def tearDown(self):
        # Clean up test data
        StrategyRequest.objects.all().delete()
        EntityModel.objects.all().delete()

    def test_model_to_entity(self):
        """Test converting from model to entity"""
        entity = EntityAdapter.model_to_entity(self.test_model, Entity)
        
        # Verify core properties
        self.assertEqual(entity.entity_id, str(self.test_uuid))
        self.assertEqual(entity.get_attributes(), self.test_attributes)
        self.assertEqual(entity.get_children(), self.test_children_ids)
        self.assertEqual(entity.get_parents(), self.test_parent_ids)
        
    def test_entity_to_new_model(self):
        """Test converting from entity to a new model"""
        # Create a test entity
        entity = Entity(entity_id=self.test_uuid)
        entity.set_attributes(self.test_attributes)
        for child_id in self.test_children_ids:
            entity.children_ids.append(child_id)
        for parent_id in self.test_parent_ids:
            entity.parent_ids.append(parent_id)
            
        # Convert to model
        model = EntityAdapter.entity_to_model(entity, model_class=EntityModel)
        
        # Verify core properties
        self.assertEqual(str(model.entity_id), self.test_uuid)
        self.assertEqual(model.attributes, self.test_attributes)
        self.assertEqual(model.children_ids, self.test_children_ids)
        self.assertEqual(model.parent_ids, self.test_parent_ids)
        self.assertEqual(model.entity_type, EntityEnum.ENTITY.value)
        self.assertEqual(model.class_path, 'shared_utils.entities.Entity.Entity')
        
    def test_entity_to_existing_model(self):
        """Test updating an existing model with entity data"""
        # Create entity with updated attributes
        entity = Entity(entity_id=self.test_uuid)
        updated_attributes = {
            'name': 'Updated Entity',
            'value': 100
        }
        entity.set_attributes(updated_attributes)
        
        # Convert to model, updating existing instance
        updated_model = EntityAdapter.entity_to_model(entity, model=self.test_model)
        
        # Verify updates
        self.assertEqual(updated_model.attributes, updated_attributes)
        self.assertEqual(str(updated_model.entity_id), self.test_uuid)
            
    def test_model_to_entity_minimal_fields(self):
        """Test conversion with minimal model fields"""
        minimal_model = EntityModel(
            entity_id=self.test_uuid,
            entity_type='ENTITY',
            attributes={},
        )
        
        entity = EntityAdapter.model_to_entity(minimal_model, Entity)
        self.assertEqual(entity.entity_id, str(self.test_uuid))
        self.assertEqual(entity.get_attributes(), {})
        self.assertEqual(entity.get_children(), [])
        self.assertEqual(entity.get_parents(), [])

        EntityModel.objects.all().delete()
        entity = EntityAdapter.model_to_entity(minimal_model)
        self.assertEqual(entity.entity_id, str(self.test_uuid))
        self.assertEqual(entity.get_attributes(), {})
        self.assertEqual(entity.get_children(), [])
        self.assertEqual(entity.get_parents(), [])


    def test_entity_to_model_defaults(self):
        """Test conversion uses EntityModel by default"""
        entity = Entity(entity_id=self.test_uuid)
        entity.set_attributes(self.test_attributes)
        
        # Convert to model without specifying model_class
        model = EntityAdapter.entity_to_model(entity)
        
        # Verify it used EntityModel and converted correctly
        self.assertIsInstance(model, EntityModel)
        self.assertEqual(model.attributes, self.test_attributes)
        self.assertEqual(str(model.entity_id), self.test_uuid)

    def test_load_strategy_requests(self):
        """Test loading strategy requests for an entity"""
        # Load strategy requests using adapter
        strategy_requests = EntityAdapter.load_strategy_requests(self.test_model)
        
        # Should only get top-level requests (not nested ones)
        self.assertEqual(len(strategy_requests), 1)  # Only the main request
        
        main_request = strategy_requests[0]
        
        # Verify main request properties
        self.assertEqual(main_request.strategy_name, "test_strategy")
        self.assertEqual(main_request.param_config, {"param1": "value1"})
        self.assertEqual(main_request.target_entity_id, self.strategy_request.target_entity_id)
        self.assertTrue(main_request.add_to_history)
        
        # Verify nested requests are loaded through the parent
        nested_requests = main_request.get_nested_requests()
        self.assertEqual(len(nested_requests), 1)
        nested_request = nested_requests[0]
        self.assertEqual(nested_request.strategy_name, "nested_strategy")
        self.assertEqual(nested_request.param_config, {"param2": "value2"})
        self.assertFalse(nested_request.add_to_history)

    def test_save_strategy_requests(self):
        """Test saving strategy requests from an entity"""
        # Create an entity with strategy requests
        entity = Entity(entity_id=self.test_uuid)
        
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = "new_strategy"
        strategy_request.param_config = {"param3": "value3"}
        strategy_request.add_to_history = True
        
        nested_request = StrategyRequestEntity()
        nested_request.strategy_name = "new_nested"
        nested_request.param_config = {"param4": "value4"}
        nested_request.add_to_history = False
        
        strategy_request.add_nested_request(nested_request)
        entity.strategy_requests.append(strategy_request)
        
        # Save using adapter
        model = EntityAdapter.entity_to_model(entity, model_class=EntityModel)
        EntityAdapter.save_strategy_requests(entity, model)
        
        # Verify requests were saved correctly
        saved_requests = StrategyRequest.objects.filter(entity_model=model)
        self.assertEqual(saved_requests.count(), 1)  # Only top-level request
        
        main_saved = saved_requests.first()
        self.assertEqual(main_saved.strategy_name, "new_strategy")
        self.assertEqual(main_saved.param_config, {"param3": "value3"})
        self.assertTrue(main_saved.add_to_history)
        
        # Verify nested request is linked to parent but not entity
        nested_saved = main_saved.nested_requests.first()
        self.assertEqual(nested_saved.strategy_name, "new_nested")
        self.assertEqual(nested_saved.param_config, {"param4": "value4"})
        self.assertFalse(nested_saved.add_to_history)
        self.assertEqual(nested_saved.parent_request, main_saved)
        self.assertIsNone(nested_saved.entity_model)

    def test_model_to_entity_with_strategy_requests(self):
        """Test that model_to_entity properly includes strategy requests"""
        entity = EntityAdapter.model_to_entity(self.test_model)
        
        # Should only get top-level requests
        self.assertEqual(len(entity.strategy_requests), 1)
        main_request = entity.strategy_requests[0]
        
        self.assertEqual(main_request.strategy_name, "test_strategy")
        self.assertEqual(main_request.param_config, {"param1": "value1"})
        
        # Check nested requests
        nested_requests = main_request.get_nested_requests()
        self.assertEqual(len(nested_requests), 1)
        nested_request = nested_requests[0]
        self.assertEqual(nested_request.strategy_name, "nested_strategy")
        self.assertEqual(nested_request.param_config, {"param2": "value2"})

    def test_entity_to_model_with_strategy_requests(self):
        """Test that entity_to_model properly saves strategy requests"""
        # Create entity with strategy requests
        entity = Entity(entity_id=self.test_uuid)
        request = StrategyRequestEntity()
        request.strategy_name = "test_strategy"
        request.param_config = {"test": "value"}
        entity.strategy_requests.append(request)
        
        # Convert to model
        model = EntityAdapter.entity_to_model(entity)
        
        # Verify strategy requests were saved
        saved_requests = StrategyRequest.objects.filter(entity_model=model)
        self.assertEqual(saved_requests.count(), 1)
        self.assertEqual(saved_requests.first().strategy_name, "test_strategy")
        self.assertEqual(saved_requests.first().param_config, {"test": "value"})

