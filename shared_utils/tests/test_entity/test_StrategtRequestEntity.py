from django.test import TestCase
from shared_utils.models import StrategyRequest
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity


class TestStrategyRequestEntity(TestCase):
    def setUp(self):
        # Create nested StrategyRequest objects for testing
        self.nested_request_1 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy1",
            strategy_path="path.to.nested_strategy_1",
            param_config={"param1": "value1"},
        )
        self.nested_request_2 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy2",
            strategy_path="path.to.nested_strategy_2",
            param_config={"param2": "value2"},
        )

        # Create the main StrategyRequest object
        self.model = StrategyRequest.objects.create(
            strategy_name="MainStrategy",
            strategy_path="path.to.main_strategy",
            param_config={"main_param": "main_value"},
        )
        # Add nested requests to the main request
        self.model.nested_requests.add(self.nested_request_1, self.nested_request_2)

    def test_from_db_with_all_fields(self):
        """
        Test that the entity is correctly populated from the model,
        including all fields and nested requests.
        """
        entity = StrategyRequestEntity.from_db(self.model)

        # Check basic attributes
        self.assertEqual(entity.id, self.model.id)
        self.assertEqual(entity.strategy_name, "MainStrategy")
        self.assertEqual(entity.param_config, {"main_param": "main_value"})

        # Check nested requests
        self.assertEqual(len(entity.nested_requests), 2)
        nested_request_1 = entity.nested_requests[0]
        self.assertEqual(nested_request_1.strategy_name, "NestedStrategy1")
        self.assertEqual(nested_request_1.param_config, {"param1": "value1"})

        nested_request_2 = entity.nested_requests[1]
        self.assertEqual(nested_request_2.strategy_name, "NestedStrategy2")
        self.assertEqual(nested_request_2.param_config, {"param2": "value2"})

        # Check timestamps
        self.assertIsNotNone(entity.created_at)
        self.assertIsNotNone(entity.updated_at)

    def test_to_db_with_updated_fields(self):
        """
        Test that an existing model instance is correctly updated from the entity.
        """
        # Load the entity from the model
        entity = StrategyRequestEntity.from_db(self.model)

        # Update the entity attributes
        entity.strategy_name = "UpdatedStrategy"
        entity.param_config = {"updated_param": "updated_value"}

        # Convert the entity back to the database model
        updated_model = entity.to_db()

        # Check that the model was updated correctly
        self.assertEqual(updated_model.id, self.model.id)  # Ensure ID is preserved
        self.assertEqual(updated_model.strategy_name, "UpdatedStrategy")
        self.assertEqual(updated_model.param_config, {"updated_param": "updated_value"})

    def test_round_trip_with_all_fields(self):
        """
        Test that converting from model to entity and back to model preserves all fields.
        """
        # Convert model to entity
        entity = StrategyRequestEntity.from_db(self.model)

        # Convert entity back to model
        updated_model = entity.to_db()

        # Ensure ID is preserved and attributes are consistent
        self.assertEqual(updated_model.id, self.model.id)
        self.assertEqual(updated_model.strategy_name, self.model.strategy_name)
        self.assertEqual(updated_model.param_config, self.model.param_config)

        # Check nested requests
        self.assertEqual(updated_model.nested_requests.count(), self.model.nested_requests.count())
        self.assertEqual(
            list(updated_model.nested_requests.values_list("strategy_name", flat=True)),
            list(self.model.nested_requests.values_list("strategy_name", flat=True)),
        )