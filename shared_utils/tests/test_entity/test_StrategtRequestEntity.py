from django.test import TestCase
from shared_utils.models import StrategyRequest
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity, StrategyRequestAdapter
from shared_utils.entities.EntityModel import EntityModel
from django.core.exceptions import ValidationError
import uuid


class TestStrategyRequestEntity(TestCase):
    def setUp(self):
        """Set up test data for StrategyRequestEntity"""
        # Create a StrategyRequestEntity instance
        self.entity_model = EntityModel.objects.create(
            entity_id=uuid.uuid4(),
            entity_type='TEST',
            attributes={}
        )
        self.entity_model.save()

        self.entity_id = str(uuid.uuid4())
        self.strategy_entity = StrategyRequestEntity(self.entity_id)
        self.strategy_entity.strategy_name = "TestStrategy"
        self.strategy_entity.param_config = {"param": "value"}
        self.strategy_entity.add_to_history = True
        self.strategy_entity.entity_model = str(self.entity_model.entity_id)

        # Create nested requests
        self.nested_request_1 = StrategyRequestEntity(str(uuid.uuid4()))
        self.nested_request_1.strategy_name = "Nested1"
        self.nested_request_1.param_config = {"nested_param1": "nested_value1"}

        self.nested_request_2 = StrategyRequestEntity(str(uuid.uuid4()))
        self.nested_request_2.strategy_name = "Nested2"
        self.nested_request_2.param_config = {"nested_param2": "nested_value2"}

        # Add nested requests
        self.strategy_entity.add_nested_request(self.nested_request_1)
        self.strategy_entity.add_nested_request(self.nested_request_2)

    def test_add_and_remove_nested_requests(self):
        """Test adding and removing nested requests"""
        nested_requests = self.strategy_entity.get_nested_requests()
        self.assertEqual(len(nested_requests), 2)

        # Remove one nested request
        self.strategy_entity.remove_nested_request(self.nested_request_1)
        nested_requests = self.strategy_entity.get_nested_requests()
        self.assertEqual(len(nested_requests), 1)
        self.assertIn(self.nested_request_2, nested_requests)
        self.assertNotIn(self.nested_request_1, nested_requests)

    def test_add_nested_request_when_nested_request_already_exists(self):
        """Test adding a nested request when it already exists should replace old request with the new request"""
        # Create a new nested request
        new_nested_request = StrategyRequestEntity(self.nested_request_1.entity_id)
        new_nested_request.strategy_name = "Nested1NEW"
        new_nested_request.param_config = {"nested_param1": "nested_value1"}

        # Add the new nested request
        self.strategy_entity.add_nested_request(new_nested_request)

        # Verify the old nested request was replaced
        nested_requests = self.strategy_entity.get_nested_requests()
        self.assertEqual(len(nested_requests), 2)
        self.assertIn(new_nested_request, nested_requests)
        self.assertNotIn(self.nested_request_1, nested_requests)
        self.assertEqual(nested_requests[1].strategy_name, "Nested1NEW")


    def test_serialize(self):
        """Test serialization of StrategyRequestEntity"""
        serialized = self.strategy_entity.serialize()
        self.assertEqual(serialized['strategy_name'], "TestStrategy")
        self.assertEqual(serialized['param_config'], {"param": "value"})
        self.assertTrue(serialized['add_to_history'])
        self.assertEqual(serialized['entity_id'], self.entity_id)
        self.assertEqual(len(serialized['nested_requests']), 2)
        self.assertEqual(serialized['nested_requests'][0]['strategy_name'], "Nested1")
        self.assertEqual(serialized['nested_requests'][1]['strategy_name'], "Nested2")

    def test_validation_rules(self):
        """Test validation rules in StrategyRequestEntity"""
        # Test invalid nested request type
        with self.assertRaises(ValueError):
            self.strategy_entity.add_nested_request("InvalidType")

    def test_to_db(self):
        """Test the to_db method for StrategyRequestEntity"""
        # Convert entity to model
        model = self.strategy_entity.to_db()

        # Verify the main model attributes
        self.assertEqual(model.strategy_name, self.strategy_entity.strategy_name)
        self.assertEqual(model.param_config, self.strategy_entity.param_config)
        self.assertEqual(model.add_to_history, self.strategy_entity.add_to_history)
        self.assertEqual(str(model.entity_model.entity_id), self.strategy_entity.entity_model)

        # Verify nested models
        nested_models = model.nested_requests.all()
        self.assertEqual(len(nested_models), 2)

        # Sort nested models for consistent verification
        nested_models = sorted(nested_models, key=lambda x: x.strategy_name)
        self.assertEqual(nested_models[0].strategy_name, "Nested1")
        self.assertEqual(nested_models[0].param_config, {"nested_param1": "nested_value1"})
        self.assertEqual(nested_models[1].strategy_name, "Nested2")
        self.assertEqual(nested_models[1].param_config, {"nested_param2": "nested_value2"})

    def test_from_db(self):
        """Test the from_db method for StrategyRequestEntity"""
        # Convert entity to model and save it to the database
        model = self.strategy_entity.to_db()
        model.save()

        # Retrieve the model and convert it back to an entity
        retrieved_model = StrategyRequest.objects.get(entity_id=model.entity_id)
        entity = StrategyRequestEntity.from_db(retrieved_model)

        # Verify the entity fields
        self.assertEqual(entity.strategy_name, self.strategy_entity.strategy_name)
        self.assertEqual(entity.param_config, self.strategy_entity.param_config)
        self.assertEqual(entity.add_to_history, self.strategy_entity.add_to_history)
        self.assertEqual(entity.entity_model, self.strategy_entity.entity_model)

        # Verify nested entities
        nested_entities = entity.get_nested_requests()
        self.assertEqual(len(nested_entities), 2)

        # Sort nested entities for consistent verification
        nested_entities = sorted(nested_entities, key=lambda x: x.strategy_name)
        self.assertEqual(nested_entities[0].strategy_name, "Nested1")
        self.assertEqual(nested_entities[0].param_config, {"nested_param1": "nested_value1"})
        self.assertEqual(nested_entities[1].strategy_name, "Nested2")
        self.assertEqual(nested_entities[1].param_config, {"nested_param2": "nested_value2"})

    def test_round_trip_to_db_and_from_db(self):
        """Test full round-trip conversion for StrategyRequestEntity"""
        # Convert entity to model and save it to the database
        model = self.strategy_entity.to_db()
        model.save()

        # Retrieve the model and convert it back to an entity
        retrieved_model = StrategyRequest.objects.get(entity_id=model.entity_id)
        round_trip_entity = StrategyRequestEntity.from_db(retrieved_model)

        # Verify the round-tripped entity matches the original
        self.assertEqual(round_trip_entity.strategy_name, self.strategy_entity.strategy_name)
        self.assertEqual(round_trip_entity.param_config, self.strategy_entity.param_config)
        self.assertEqual(round_trip_entity.add_to_history, self.strategy_entity.add_to_history)
        self.assertEqual(round_trip_entity.entity_model, self.strategy_entity.entity_model)

        # Verify nested requests
        original_nested = sorted(self.strategy_entity.get_nested_requests(), key=lambda x: x.strategy_name)
        round_trip_nested = sorted(round_trip_entity.get_nested_requests(), key=lambda x: x.strategy_name)

        self.assertEqual(len(original_nested), len(round_trip_nested))
        for original, round_trip in zip(original_nested, round_trip_nested):
            self.assertEqual(original.strategy_name, round_trip.strategy_name)
            self.assertEqual(original.param_config, round_trip.param_config)


class TestStrategyRequestAdapter(TestCase):
    def setUp(self):
        """Set up test data for StrategyRequestAdapter"""
        # Create an EntityModel instance
        self.entity_model = EntityModel.objects.create(
            entity_id=uuid.uuid4(),
            entity_type='TEST',
            attributes={}
        )

        # Create the main StrategyRequest model (top-level)
        self.model = StrategyRequest.objects.create(
            entity_id=uuid.uuid4(),
            strategy_name="MainStrategy",
            param_config={"main_param": "main_value"},
            entity_model=self.entity_model,
            add_to_history=True
        )

        # Create child StrategyRequest models (nested requests)
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

    def test_model_to_entity(self):
        """Test converting a StrategyRequest model to a StrategyRequestEntity"""
        entity = StrategyRequestAdapter.model_to_entity(self.model)

        # Check basic attributes
        self.assertEqual(str(entity.entity_id), str(self.model.entity_id))
        self.assertEqual(entity.strategy_name, "MainStrategy")
        self.assertEqual(entity.param_config, {"main_param": "main_value"})
        self.assertEqual(str(entity.entity_model), str(self.entity_model.entity_id))

        # Check nested requests
        nested_requests = entity.get_nested_requests()
        self.assertEqual(len(nested_requests), 2)

        # Sort nested requests for consistent testing
        nested_requests.sort(key=lambda x: x.strategy_name)

        # Verify nested request attributes
        self.assertEqual(nested_requests[0].strategy_name, "NestedStrategy1")
        self.assertEqual(nested_requests[0].param_config, {"param1": "value1"})
        self.assertEqual(nested_requests[1].strategy_name, "NestedStrategy2")
        self.assertEqual(nested_requests[1].param_config, {"param2": "value2"})

    def test_entity_to_model(self):
        """Test converting a StrategyRequestEntity to a StrategyRequest model"""
        # Create an entity
        entity = StrategyRequestEntity(str(uuid.uuid4()))
        entity.strategy_name = "NewStrategy"
        entity.param_config = {"param": "value"}
        entity.add_to_history = True
        entity.entity_model = str(self.entity_model.entity_id)

        # Add nested requests
        nested = StrategyRequestEntity(str(uuid.uuid4()))
        nested.strategy_name = "NestedNew"
        nested.param_config = {"nested_param": "nested_value"}
        entity.add_nested_request(nested)

        # Convert to model
        model = StrategyRequestAdapter.entity_to_model(entity)

        # Verify the model
        self.assertEqual(model.strategy_name, "NewStrategy")
        self.assertEqual(model.param_config, {"param": "value"})
        self.assertEqual(str(model.entity_model.entity_id), str(self.entity_model.entity_id))

        # Verify nested request
        nested_model = model.nested_requests.first()
        self.assertEqual(nested_model.strategy_name, "NestedNew")
        self.assertEqual(nested_model.param_config, {"nested_param": "nested_value"})

    def test_round_trip_conversion(self):
        """Test full round-trip conversion from model to entity and back"""
        # Convert model to entity
        entity = StrategyRequestAdapter.model_to_entity(self.model)

        # Convert back to model
        model = StrategyRequestAdapter.entity_to_model(entity)

        # Verify the resulting model matches the original
        self.assertEqual(model.strategy_name, self.model.strategy_name)
        self.assertEqual(model.param_config, self.model.param_config)
        self.assertEqual(str(model.entity_model.entity_id), str(self.entity_model.entity_id))

        # Verify nested requests
        nested_models = model.nested_requests.all()
        self.assertEqual(len(nested_models), 2)

        # Sort for consistent testing
        nested_models = sorted(nested_models, key=lambda x: x.strategy_name)

        self.assertEqual(nested_models[0].strategy_name, "NestedStrategy1")
        self.assertEqual(nested_models[0].param_config, {"param1": "value1"})
        self.assertEqual(nested_models[1].strategy_name, "NestedStrategy2")
        self.assertEqual(nested_models[1].param_config, {"param2": "value2"})