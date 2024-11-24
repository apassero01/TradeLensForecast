from enum import Enum

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from training_session.models import TrainingSession

class TrainingSessionStatus(Enum):
    ACTIVE = 1
    INACTIVE = 2

class TrainingSessionEntity(Entity):
    entity_name = EntityEnum.TRAINING_SESSION
    def __init__(self):
        super().__init__()
        self.X_features = None
        self.y_features = None
        self.status = TrainingSessionStatus.ACTIVE
        self.start_date = None
        self.end_date = None
        self.created_at = None
        self.X_feature_dict = None
        self.y_feature_dict = None
        self.strategy_history = []

    @staticmethod
    def get_maximum_members():
        return{
            EntityEnum.MODEL_STAGE: None,
        }

    def to_db(self, model=None):
        """
        Converts the entity into a Django model instance using the adapter.
        """
        return TrainingSessionAdapter.entity_to_model(self, model)

    @classmethod
    def from_db(cls, model):
        """
        Creates an entity from a Django model instance using the adapter.
        """
        return TrainingSessionAdapter.model_to_entity(model)

class TrainingSessionAdapter:
    @staticmethod
    def model_to_entity(model: TrainingSession) -> TrainingSessionEntity:
        """
        Converts a TrainingSession Django model instance to a TrainingSessionEntity.
        """
        entity = TrainingSessionEntity()
        entity.id = model.pk  # Track the database model's ID
        entity.X_features = model.X_features
        entity.y_features = model.y_features
        entity.status = model.status
        entity.start_date = model.start_date
        entity.end_date = model.end_date
        entity.created_at = model.created_at
        entity.X_feature_dict = model.X_feature_dict
        entity.y_feature_dict = model.y_feature_dict
        entity.strategy_history = model.strategy_history
        return entity

    @staticmethod
    def entity_to_model(entity: TrainingSessionEntity, model: TrainingSession = None) -> TrainingSession:
        """
        Converts a TrainingSessionEntity to a TrainingSession Django model instance.
        If `model` is not provided, creates a new instance.
        """
        if model is None:
            # If the entity has an ID, fetch the existing model
            if entity.id is not None:
                model = TrainingSession.objects.get(id=entity.id)
            else:
                model = TrainingSession()  # Create a new instance

        model.X_features = entity.X_features
        model.y_features = entity.y_features
        model.status = entity.status
        model.start_date = entity.start_date
        model.end_date = entity.end_date
        model.created_at = entity.created_at  # Typically auto-managed by DB
        model.X_feature_dict = entity.X_feature_dict
        model.y_feature_dict = entity.y_feature_dict
        model.strategy_history = entity.strategy_history
        return model

