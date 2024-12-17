from sequenceset_manager.models import SequenceSet
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from typing import Optional

class SequenceSetEntity(Entity):
    entity_name = EntityEnum.SEQUENCE_SET

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.id = None
        self.dataset_type= None
        self.start_timestamp = None
        self.end_timestamp = None
        self.sequence_length = None
        self.feature_dict = None
        self.metadata = None
        self.sequences = None


    def to_db(self):
        raise NotImplementedError("Child classes must implement the 'to_db' method.")
    @classmethod
    def from_db(cls, data):
        return SequenceSetAdapter.model_to_entity(data)

    @staticmethod
    def get_maximum_members():
        return {
            EntityEnum.SEQUENCE_SET: None,
        }

    def serialize(self):
        return {
            'entity_name': self.entity_name.value,
            'children': [child.serialize() for child in self.children],
            'meta_data': {
                'ticker': self.get_attribute('metadata')['ticker'],
                'start': self.get_attribute('start_timestamp'),
                'sequence_length': self.get_attribute('sequence_length'),
                'X_features': self.get_attribute('X_features'),
                'y_features': self.get_attribute('y_features'),
            },
            'path': self.path

        }


class SequenceSetAdapter:
    @staticmethod
    def model_to_entity(model: SequenceSet) -> SequenceSetEntity:
        """
        Converts a TrainingSession Django model instance to a TrainingSessionEntity.
        """
        entity = SequenceSetEntity()
        # entity.id = model.pk
        # entity.dataset_type = model.dataset_type
        # entity.start_timestamp = model.start_timestamp
        # entity.end_timestamp = model.end_timestamp
        # entity.sequence_length = model.sequence_length
        # entity.feature_dict = model.feature_dict
        # entity.metadata = model.metadata
        entity.set_attribute('id', model.pk)
        entity.set_attribute('dataset_type', model.dataset_type)
        entity.set_attribute('start_timestamp', model.start_timestamp)
        entity.set_attribute('end_timestamp', model.end_timestamp)
        entity.set_attribute('sequence_length', model.sequence_length)
        entity.set_attribute('feature_dict', model.feature_dict)
        entity.set_attribute('metadata', model.metadata)

        return entity