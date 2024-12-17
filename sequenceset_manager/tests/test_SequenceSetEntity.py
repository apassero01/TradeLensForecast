from django.test import TestCase

from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from sequenceset_manager.models import SequenceSet


class TestSequenceSetEntity(TestCase):
    def setUp(self):
        # Create a model instance for testing
        self.model = SequenceSet.objects.create(
            dataset_type="stock",
            start_timestamp="2023-01-01T00:00:00Z",
            end_timestamp="2023-01-10T00:00:00Z",
            sequence_length=10,
            feature_dict={"feature1": "value1", "feature2": "value2"},
            metadata={"key": "value"}
        )

    def test_from_db_with_all_fields(self):
        """
        Test that the entity is correctly populated from the model,
        including all fields.
        """
        entity = SequenceSetEntity.from_db(self.model)

        # Check that all attributes, including ID and new fields, are correctly populated
        self.assertEqual(entity.get_attribute('id'), self.model.id)
        self.assertEqual(entity.get_attribute('dataset_type'), self.model.dataset_type)
        self.assertEqual(entity.get_attribute('start_timestamp'), self.model.start_timestamp)
        self.assertEqual(entity.get_attribute('end_timestamp'), self.model.end_timestamp)
        self.assertEqual(entity.get_attribute('sequence_length'), self.model.sequence_length)
        self.assertEqual(entity.get_attribute('feature_dict'), self.model.feature_dict)
        self.assertEqual(entity.get_attribute('metadata'), self.model.metadata)
