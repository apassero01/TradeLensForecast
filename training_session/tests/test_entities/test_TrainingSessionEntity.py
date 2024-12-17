from django.test import TestCase
from training_session.models import TrainingSession
from shared_utils.models import StrategyRequest
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity


class TestTrainingSessionEntity(TestCase):
    def setUp(self):
        """
        Set up a TrainingSession and related StrategyRequests for testing.
        """
        # Create a TrainingSession model instance for testing
        self.model = TrainingSession.objects.create(
            entity_id="session_123",
        )

        # Create top-level StrategyRequests linked to the TrainingSession
        self.strategy_request_1 = StrategyRequest.objects.create(
            strategy_name="TopLevelStrategy1",
            strategy_path="path.to.strategy1",
            param_config={"param1": "value1"},
            training_session=self.model,
            parent_request=None  # This is a top-level request
        )

        self.strategy_request_2 = StrategyRequest.objects.create(
            strategy_name="TopLevelStrategy2",
            strategy_path="path.to.strategy2",
            param_config={"param2": "value2"},
            training_session=self.model,
            parent_request=None  # This is also a top-level request
        )

        # Create nested StrategyRequests linked to the top-level requests
        self.nested_request_1 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy1",
            strategy_path="path.to.nested_strategy_1",
            param_config={"param_nested1": "nested_value1"},
            parent_request=self.strategy_request_1
        )

        self.nested_request_2 = StrategyRequest.objects.create(
            strategy_name="NestedStrategy2",
            strategy_path="path.to.nested_strategy_2",
            param_config={"param_nested2": "nested_value2"},
            parent_request=self.strategy_request_2
        )

    def test_from_db_with_all_fields(self):
        """
        Test that the entity is correctly populated from the model,
        including all fields and strategy history.
        """
        entity = TrainingSessionEntity.from_db(self.model)

        # Check that the entity contains the correct fields
        self.assertEqual(entity.id, self.model.id)
        self.assertEqual(entity.created_at, self.model.created_at)

        # Check strategy history (should only contain top-level requests)
        self.assertEqual(len(entity.strategy_history), 2)
        
        strategy_names = [s.strategy_name for s in entity.strategy_history]
        self.assertIn("TopLevelStrategy1", strategy_names)
        self.assertIn("TopLevelStrategy2", strategy_names)

        # Ensure nested requests are NOT included in the top-level strategy history
        self.assertNotIn("NestedStrategy1", strategy_names)
        self.assertNotIn("NestedStrategy2", strategy_names)

    def test_to_db_with_updated_fields(self):
        """
        Test that an existing model instance is correctly updated from the entity,
        including updates to strategy histories.
        """
        # Load the entity from the model
        entity = TrainingSessionEntity.from_db(self.model)

        # Add a new strategy request to the strategy history
        new_strategy_entity = StrategyRequestEntity()
        new_strategy_entity.strategy_name = "NewStrategy"
        new_strategy_entity.strategy_path = "path.to.new_strategy"
        new_strategy_entity.param_config = {"new_param": "new_value"}
        entity.add_to_strategy_history(new_strategy_entity)

        # Convert the entity back to the database model
        updated_model = TrainingSessionEntity.to_db(entity)

        # Check that the model was updated correctly
        self.assertEqual(updated_model.id, self.model.id)
        self.assertEqual(updated_model.created_at, self.model.created_at)

        # Check that the new strategy request was saved and linked to the updated TrainingSession
        self.assertTrue(StrategyRequest.objects.filter(
            strategy_name="NewStrategy", 
            training_session=updated_model
        ).exists())

    def test_round_trip_with_all_fields(self):
        """
        Test that converting from model to entity and back to model preserves all fields and relationships.
        """
        # Convert model to entity
        entity = TrainingSessionEntity.from_db(self.model)

        # Convert entity back to model
        updated_model = TrainingSessionEntity.to_db(entity)

        # Ensure ID is preserved and attributes are consistent
        self.assertEqual(updated_model.id, self.model.id)

        # Check that all strategy histories were preserved
        strategy_requests = updated_model.strategy_requests.filter(parent_request__isnull=True)
        self.assertEqual(strategy_requests.count(), 2)
        
        strategy_names = strategy_requests.values_list('strategy_name', flat=True)
        self.assertIn("TopLevelStrategy1", strategy_names)
        self.assertIn("TopLevelStrategy2", strategy_names)

    def test_add_to_strategy_history(self):
        """
        Test that adding a StrategyRequestEntity to the strategy history works as expected.
        """
        entity = TrainingSessionEntity()

        strategy_entity = StrategyRequestEntity()
        strategy_entity.strategy_name = "NewStrategy"
        strategy_entity.strategy_path = "path.to.new_strategy"
        strategy_entity.param_config = {"param1": "value1"}

        entity.add_to_strategy_history(strategy_entity)

        # Check that strategy history was updated
        self.assertEqual(len(entity.strategy_history), 1)
        self.assertEqual(entity.strategy_history[0].strategy_name, "NewStrategy")