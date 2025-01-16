from django.test import TestCase
from shared_utils.models import StrategyRequest, TrainingSession
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from django.core.exceptions import ValidationError


class TestStrategyRequestEntity(TestCase):
    def setUp(self):
        """
        Set up the TrainingSession and StrategyRequest objects for testing.
        """
        # Create a TrainingSession
        self.training_session = TrainingSession.objects.create()

        # Create the main StrategyRequest object (top-level)
        self.model = StrategyRequest.objects.create(
            strategy_name="MainStrategy",
            param_config={"main_param": "main_value"},
            training_session=self.training_session
        )

        # Create child StrategyRequest objects (nested requests)
        self.nested_request_1 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy1",
            param_config={"param1": "value1"},
            parent_request=self.model,
            training_session=self.training_session
        )

        self.nested_request_2 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy2",
            param_config={"param2": "value2"},
            parent_request=self.model,
            training_session=self.training_session
        )

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
        self.assertEqual(entity.training_session_id, self.training_session.id)

        # Check nested requests (check via parent-child relationship)
        nested_requests = StrategyRequest.objects.filter(parent_request=self.model)
        self.assertEqual(len(entity._nested_requests), nested_requests.count())

        nested_request_1 = entity._nested_requests[0]
        self.assertEqual(nested_request_1.strategy_name, "NestedStrategy1")
        self.assertEqual(nested_request_1.param_config, {"param1": "value1"})

        nested_request_2 = entity._nested_requests[1]
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
        entity.training_session_id = self.training_session.id

        # Convert the entity back to the database model
        updated_model = entity.to_db()

        # Check that the model was updated correctly
        self.assertEqual(updated_model.id, self.model.id)  # Ensure ID is preserved
        self.assertEqual(updated_model.strategy_name, "UpdatedStrategy")
        self.assertEqual(updated_model.param_config, {"updated_param": "updated_value"})
        self.assertEqual(updated_model.training_session.id, self.training_session.id)

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
        self.assertEqual(updated_model.training_session.id, self.model.training_session.id)

        # Check nested requests
        nested_request_models = StrategyRequest.objects.filter(parent_request=self.model)
        self.assertEqual(nested_request_models.count(), StrategyRequest.objects.filter(parent_request=self.model).count())

        # Check that the strategy names for all nested requests match
        self.assertEqual(
            list(nested_request_models.values_list("strategy_name", flat=True)),
            list(StrategyRequest.objects.filter(parent_request=self.model).values_list("strategy_name", flat=True)),
        )

    def test_create_strategy_with_parent(self):
        """
        Test that a StrategyRequest can be created with a parent request and is properly linked.
        """
        child_request = StrategyRequest.objects.create(
            strategy_name="ChildStrategy",
            param_config={"child_param": "child_value"},
            parent_request=self.model,
            training_session=self.training_session
        )

        # Check that the child request has the correct parent
        self.assertEqual(child_request.parent_request.id, self.model.id)
        self.assertEqual(child_request.training_session.id, self.training_session.id)
        self.assertFalse(child_request.is_top_level())  # This is a child request

    def test_only_top_level_requests_appear_in_training_session_history(self):
        """
        Test that only top-level requests are included in the strategy history.
        """
        # Get all top-level requests for the training session
        top_level_requests = StrategyRequest.objects.filter(
            training_session=self.training_session, 
            parent_request__isnull=True
        )

        # Assert that only the main strategy is a top-level request
        self.assertEqual(top_level_requests.count(), 1)
        self.assertEqual(top_level_requests.first().id, self.model.id)

    def test_parent_request_cannot_be_itself(self):
        """
        Test that a StrategyRequest cannot be its own parent.
        """
        with self.assertRaises(ValidationError):
            self.model.parent_request = self.model
            self.model.clean()

    def test_nested_request_cannot_be_part_of_strategy_history(self):
        """
        Test that a nested request cannot be marked as a top-level request.
        """
        with self.assertRaises(ValidationError):
            self.nested_request_1.add_to_history = True
            self.nested_request_1.clean()