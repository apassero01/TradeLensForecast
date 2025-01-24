from django.test import TestCase
from shared_utils.models import StrategyRequest
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.EntityModel import EntityModel
from django.core.exceptions import ValidationError
import uuid


class TestStrategyRequestEntity(TestCase):
    def setUp(self):
        """Set up test data"""
        # Create an EntityModel instance
        self.entity_model = EntityModel.objects.create(
            entity_id=uuid.uuid4(),
            entity_type='TEST',
            attributes={}
        )

        # Create the main StrategyRequest object (top-level)
        self.model = StrategyRequest.objects.create(
            entity_id=uuid.uuid4(),
            strategy_name="MainStrategy",
            param_config={"main_param": "main_value"},
            entity_model=self.entity_model,
            add_to_history=True
        )

        # Create child StrategyRequest objects (nested requests)
        self.nested_request_1 = StrategyRequest.objects.create(
            entity_id=uuid.uuid4(),
            strategy_name="NestedStrategy1",
            param_config={"param1": "value1"},
            parent_request=self.model,
            add_to_history=False
        )

        self.nested_request_2 = StrategyRequest.objects.create(
            entity_id=uuid.uuid4(),
            strategy_name="NestedStrategy2",
            param_config={"param2": "value2"},
            parent_request=self.model,
            add_to_history=False
        )

    def tearDown(self):
        """Clean up test data"""
        StrategyRequest.objects.all().delete()
        EntityModel.objects.all().delete()

    def test_from_db_with_all_fields(self):
        """Test that the entity is correctly populated from the model"""
        entity = StrategyRequestEntity.from_db(self.model)

        # Check basic attributes
        self.assertEqual(str(entity.entity_id), str(self.model.entity_id))
        self.assertEqual(entity.strategy_name, "MainStrategy")
        self.assertEqual(entity.param_config, {"main_param": "main_value"})
        self.assertEqual(entity.entity_model, str(self.entity_model.entity_id))

        # Check nested requests
        nested_requests = entity.get_nested_requests()
        self.assertEqual(len(nested_requests), 2)

        # Sort nested requests by strategy name for consistent testing
        nested_requests.sort(key=lambda x: x.strategy_name)
        
        self.assertEqual(nested_requests[0].strategy_name, "NestedStrategy1")
        self.assertEqual(nested_requests[0].param_config, {"param1": "value1"})
        self.assertEqual(nested_requests[1].strategy_name, "NestedStrategy2")
        self.assertEqual(nested_requests[1].param_config, {"param2": "value2"})

        # Check timestamps
        self.assertIsNotNone(entity.created_at)
        self.assertIsNotNone(entity.updated_at)

    def test_to_db_with_updated_fields(self):
        """Test that an existing model is correctly updated from the entity"""
        entity = StrategyRequestEntity(str(uuid.uuid4()))
        entity.strategy_name = "UpdatedStrategy"
        entity.param_config = {"updated_param": "updated_value"}
        entity.add_to_history = True
        entity.entity_model = str(self.entity_model.entity_id)

        # Convert to model
        model = entity.to_db()

        # Check that the model was updated correctly
        self.assertEqual(str(model.entity_id), str(entity.entity_id))
        self.assertEqual(model.strategy_name, "UpdatedStrategy")
        self.assertEqual(model.param_config, {"updated_param": "updated_value"})
        self.assertEqual(str(model.entity_model.entity_id), str(self.entity_model.entity_id))

    def test_nested_requests_handling(self):
        """Test adding and retrieving nested requests"""
        parent = StrategyRequestEntity(str(uuid.uuid4()))
        child1 = StrategyRequestEntity(str(uuid.uuid4()))
        child2 = StrategyRequestEntity(str(uuid.uuid4()))

        # Add nested requests
        parent.add_nested_request(child1)
        parent.add_nested_request(child2)

        # Check nested requests
        nested = parent.get_nested_requests()
        self.assertEqual(len(nested), 2)
        self.assertIn(child1, nested)
        self.assertIn(child2, nested)

        # Remove a nested request
        parent.remove_nested_request(child1)
        nested = parent.get_nested_requests()
        self.assertEqual(len(nested), 1)
        self.assertNotIn(child1, nested)
        self.assertIn(child2, nested)

    def test_validation_rules(self):
        """Test model validation rules"""
        # Test self-referential parent
        with self.assertRaises(ValidationError):
            self.model.parent_request = self.model
            self.model.clean()

        # Test nested request with add_to_history=True
        with self.assertRaises(ValidationError):
            self.nested_request_1.add_to_history = True
            self.nested_request_1.clean()

    def test_serialize(self):
        """Test entity serialization"""
        entity = StrategyRequestEntity(str(uuid.uuid4()))
        entity.strategy_name = "TestStrategy"
        entity.param_config = {"test": "value"}
        entity.add_to_history = True
        entity.target_entity_id = str(uuid.uuid4())

        nested = StrategyRequestEntity(str(uuid.uuid4()))
        nested.strategy_name = "NestedStrategy"
        nested.param_config = {"nested": "value"}
        entity.add_nested_request(nested)

        serialized = entity.serialize()

        self.assertEqual(serialized['strategy_name'], "TestStrategy")
        self.assertEqual(serialized['param_config'], {"test": "value"})
        self.assertTrue(serialized['add_to_history'])
        self.assertEqual(serialized['target_entity_id'], entity.target_entity_id)
        self.assertEqual(len(serialized['nested_requests']), 1)
        self.assertEqual(serialized['nested_requests'][0]['strategy_name'], "NestedStrategy")