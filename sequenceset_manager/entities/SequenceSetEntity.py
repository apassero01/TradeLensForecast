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


    @staticmethod
    def get_maximum_members():
        return {
            EntityEnum.SEQUENCE_SET: None,
        }

    def serialize(self):
        sup_dict = super().serialize()
        sup_dict['meta_data'] = {
            'ticker': self.get_attribute('metadata')['ticker'] if self.has_attribute('metadata') else None,
            'start': self.get_attribute('start_timestamp') if self.has_attribute('start_timestamp') else None,
            'sequence_length': self.get_attribute('sequence_length') if self.has_attribute('sequence_length') else None,
            'X_features': self.get_attribute('X_features') if self.has_attribute('X_features') else None,
            'y_features': self.get_attribute('y_features') if self.has_attribute('y_features') else None,
        }
        return sup_dict


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