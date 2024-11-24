from django.test import TestCase

from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.models import TrainingSession


class TestTrainingSessionEntity(TestCase):
    def setUp(self):
        # Create a model instance for testing
        self.model = TrainingSession.objects.create(
            X_features={"feature1": "value1"},
            y_features={"feature2": "value2"},
            status=1,
            start_date="2024-01-01",
            end_date="2024-12-31",
            X_feature_dict={"dict_key1": "dict_value1"},
            y_feature_dict={"dict_key2": "dict_value2"},
            ordered_model_set_strategies=["strategy1", "strategy2"]
        )

    def test_from_db_with_all_fields(self):
        """
        Test that the entity is correctly populated from the model,
        including all fields.
        """
        entity = TrainingSessionEntity.from_db(self.model)

        # Check that all attributes, including ID and new fields, are correctly populated
        self.assertEqual(entity.id, self.model.id)
        self.assertEqual(entity.X_features, {"feature1": "value1"})
        self.assertEqual(entity.y_features, {"feature2": "value2"})
        self.assertEqual(entity.status, 1)
        self.assertEqual(str(entity.start_date), "2024-01-01")
        self.assertEqual(str(entity.end_date), "2024-12-31")
        self.assertEqual(entity.X_feature_dict, {"dict_key1": "dict_value1"})
        self.assertEqual(entity.y_feature_dict, {"dict_key2": "dict_value2"})
        self.assertEqual(entity.ordered_model_set_strategies, ["strategy1", "strategy2"])


    def test_to_db_with_updated_fields(self):
        """
        Test that an existing model instance is correctly updated from the entity.
        """
        # Load the entity from the model
        entity = TrainingSessionEntity.from_db(self.model)

        # Update the entity attributes
        entity.X_features = {"new_feature": "new_value"}
        entity.status = 2
        entity.X_feature_dict = {"updated_key1": "updated_value1"}
        entity.ordered_model_set_strategies = ["updated_strategy"]

        # Convert the entity back to the database model
        updated_model = entity.to_db(entity)

        # Check that the model was updated correctly
        self.assertEqual(updated_model.id, self.model.id)  # Ensure ID is preserved
        self.assertEqual(updated_model.X_features, {"new_feature": "new_value"})
        self.assertEqual(updated_model.status, 2)
        self.assertEqual(updated_model.X_feature_dict, {"updated_key1": "updated_value1"})
        self.assertEqual(updated_model.ordered_model_set_strategies, ["updated_strategy"])


    def test_round_trip_with_all_fields(self):
        """
        Test that converting from model to entity and back to model preserves all fields.
        """
        # Convert model to entity
        entity = TrainingSessionEntity.from_db(self.model)

        # Convert entity back to model
        updated_model = entity.to_db()

        # Ensure ID is preserved and attributes are consistent
        self.assertEqual(updated_model.id, self.model.id)
        self.assertEqual(updated_model.X_features, self.model.X_features)
        self.assertEqual(updated_model.y_features, self.model.y_features)
        self.assertEqual(updated_model.status, self.model.status)
        self.assertEqual(updated_model.X_feature_dict, self.model.X_feature_dict)
        self.assertEqual(updated_model.y_feature_dict, self.model.y_feature_dict)
        self.assertEqual(updated_model.ordered_model_set_strategies, self.model.ordered_model_set_strategies)