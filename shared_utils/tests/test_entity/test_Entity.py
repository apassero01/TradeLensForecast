from django.test import TestCase
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
import numpy as np

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

