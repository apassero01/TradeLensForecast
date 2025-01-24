import uuid
from django.test import TestCase
from shared_utils.entities.Entity import Entity, EntityAdapter
from shared_utils.entities.EntityModel import EntityModel
from shared_utils.entities.EnityEnum import EntityEnum

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
            id=self.test_uuid,
            entity_type='ENTITY',
            attributes=self.test_attributes,
            children_ids=self.test_children_ids,
            parent_ids=self.test_parent_ids,
            class_path='shared_utils.entities.Entity'
        )
        
    def test_model_to_entity_conversion(self):
        """Test converting from model to entity"""
        entity = Entity.from_db(self.test_model)
        
        # Verify core properties
        self.assertEqual(entity.entity_id, str(self.test_uuid))
        self.assertEqual(entity.get_attributes(), self.test_attributes)
        self.assertEqual(entity.get_children(), self.test_children_ids)
        self.assertEqual(entity.get_parents(), self.test_parent_ids)
        
    def test_entity_to_model_conversion_new(self):
        """Test converting from entity to a new model"""
        # Create a test entity
        entity = Entity(entity_id=self.test_uuid)
        entity.set_attributes(self.test_attributes)
        for child_id in self.test_children_ids:
            entity.children_ids.append(child_id)
        for parent_id in self.test_parent_ids:
            entity.parent_ids.append(parent_id)
            
        # Convert to model
        model = entity.to_db(model_class=EntityModel)
        
        # Verify core properties
        self.assertEqual(str(model.id), self.test_uuid)
        self.assertEqual(model.attributes, self.test_attributes)
        self.assertEqual(model.children_ids, self.test_children_ids)
        self.assertEqual(model.parent_ids, self.test_parent_ids)
        self.assertEqual(model.entity_type, EntityEnum.ENTITY.value)
        self.assertEqual(model.class_path, 'shared_utils.entities.Entity')
        
    def test_entity_to_model_conversion_existing(self):
        """Test converting from entity to an existing model"""
        # Create and save a test model
        self.test_model.save()
        
        # Create entity with updated attributes
        entity = Entity(entity_id=self.test_uuid)
        updated_attributes = {
            'name': 'Updated Entity',
            'value': 100
        }
        entity.set_attributes(updated_attributes)
        
        # Convert to model, updating existing instance
        updated_model = entity.to_db(model=self.test_model)
        
        # Verify updates
        self.assertEqual(updated_model.attributes, updated_attributes)
        self.assertEqual(str(updated_model.id), self.test_uuid)
        
    def test_invalid_entity_id(self):
        """Test handling of invalid entity ID"""
        with self.assertRaises(ValueError):
            Entity(entity_id="invalid-uuid")
            
    def test_model_to_entity_without_optional_fields(self):
        """Test converting a minimal model to entity"""
        minimal_model = EntityModel(
            id=self.test_uuid,
            entity_type='ENTITY',
            attributes={},
        )
        
        entity = Entity.from_db(minimal_model)
        self.assertEqual(entity.entity_id, str(self.test_uuid))
        self.assertEqual(entity.get_attributes(), {})
        self.assertEqual(entity.get_children(), [])
        self.assertEqual(entity.get_parents(), [])

    def test_entity_to_model_without_model_or_class(self):
        """Test that conversion fails when neither model nor model_class is provided"""
        entity = Entity(entity_id=self.test_uuid)
        with self.assertRaises(ValueError):
            entity.to_db()
