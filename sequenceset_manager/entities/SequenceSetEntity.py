from sequenceset_manager.models import SequenceSet
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity


class SequenceSetEntity(Entity):
    entity_name = EntityEnum.SEQUENCE_SETS

    def __init__(self):
        super().__init__()
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
            EntityEnum.SEQUENCE_SETS: None,
        }


class SequenceSetAdapter:
    @staticmethod
    def model_to_entity(model: SequenceSet) -> SequenceSetEntity:
        """
        Converts a TrainingSession Django model instance to a TrainingSessionEntity.
        """
        entity = SequenceSetEntity()
        entity.id = model.pk
        entity.dataset_type = model.dataset_type
        entity.start_timestamp = model.start_timestamp
        entity.end_timestamp = model.end_timestamp
        entity.sequence_length = model.sequence_length
        entity.feature_dict = model.feature_dict
        entity.metadata = model.metadata

        return entity