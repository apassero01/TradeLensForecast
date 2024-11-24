from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.models import StrategyRequest


class StrategyRequestEntity(Entity):
    entity_name = EntityEnum.ENTITY

    def __init__(self):
        super().__init__()
        self.strategy_name = None
        self.strategy_path = None
        self.param_config = None
        self.nested_requests = None
        self.created_at = None
        self.updated_at = None
        self.id = None
        self.ret_val = {}
        self.is_applied = False # Flag to indicate if the strategy has been applied

    def to_db(self):
        return StrategyRequestAdapter.entity_to_model(self)

    @classmethod
    def from_db(cls, data):
        return StrategyRequestAdapter.model_to_entity(data)


class StrategyRequestAdapter:
    @staticmethod
    def model_to_entity(model: StrategyRequest) -> StrategyRequestEntity:
        """
        Converts a TrainingSession Django model instance to a TrainingSessionEntity.
        """
        entity = StrategyRequestEntity()
        entity.id = model.pk  # Track the database model's ID
        entity.strategy_name = model.strategy_name
        entity.param_config = model.param_config
        entity.nested_requests = model.nested_requests.all()

        entity.created_at = model.created_at
        entity.updated_at = model.updated_at

        return entity

    @staticmethod
    def entity_to_model(entity: StrategyRequestEntity, model: StrategyRequest = None) -> StrategyRequest:
        """
        Converts a TrainingSessionEntity to a TrainingSession Django model instance.
        If `model` is not provided, creates a new instance.
        """
        if model is None:
            # If the entity has an ID, fetch the existing model
            if entity.id is not None:
                model = StrategyRequest.objects.get(id=entity.id)
            else:
                model = StrategyRequest(strategy_name=entity.strategy_name,
                                        param_config=entity.param_config,
                                        nested_requests=entity.nested_requests,
                                        created_at=entity.created_at,
                                        updated_at=entity.updated_at)
        return model


