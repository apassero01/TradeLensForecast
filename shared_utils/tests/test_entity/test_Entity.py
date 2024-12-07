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
        self.child1.set_attribute('test_data', {
            'X_train': np.array([[1, 2], [3, 4]]),
            'y_train': np.array([1, 0])
        })
        
        self.child2.set_attribute('test_data', {
            'X_train': np.array([[5, 6], [7, 8]]),
            'y_train': np.array([0, 1])
        })

    def test_entity_path_generation(self):
        """Test that entity paths are correctly generated with entity names and UUIDs"""
        # Add entities and capture paths
        self.parent.add_child(self.child1)
        
        # Verify paths are correctly formatted
        expected_path = (
            f"{EntityEnum.ENTITY.value}:{self.parent.entity_id}/"
            f"{EntityEnum.DATA_BUNDLE.value}:{self.child1.entity_id}"
        )
        self.assertEqual(self.child1.path, expected_path)

    def test_root_entity_path(self):
        """Test that root entity path is correctly formatted"""
        expected_path = f"{EntityEnum.ENTITY.value}:{self.parent.entity_id}"
        self.assertEqual(self.parent.path, expected_path)

    def test_find_entity_by_path(self):
        """Test finding entities by their path"""
        self.parent.add_child(self.child1)
        
        # Find entity by path
        path = self.child1.path
        found_entity = self.parent.find_entity_by_path(path)
        self.assertEqual(found_entity, self.child1)

    def test_find_entities_by_paths(self):
        """Test finding multiple entities by their paths"""
        # Build entity hierarchy
        self.parent.add_child(self.child1)
        self.parent.add_child(self.child2)
        
        # Find multiple entities
        paths = [self.child1.path, self.child2.path]
        found_entities = self.parent.find_entities_by_paths(paths)
        
        self.assertEqual(found_entities[self.child1.path], self.child1)
        self.assertEqual(found_entities[self.child2.path], self.child2)

    def test_path_prefix_optimization(self):
        """Test that find_entities_by_paths skips branches that can't contain target paths"""
        self.parent.add_child(self.child1)
        self.parent.add_child(self.child2)
        
        # Try to find path in child2 (should skip child1 branch)
        target_path = f"{EntityEnum.ENTITY.value}:{self.parent.entity_id}/" \
                     f"{EntityEnum.DATA_BUNDLE.value}:nonexistent-uuid"
                     
        results = self.parent.find_entities_by_paths([target_path])
        self.assertIsNone(results[target_path])

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
        
        self.assertEqual(self.child1._parent, self.parent)
        self.assertIn(self.child1, self.parent.children)