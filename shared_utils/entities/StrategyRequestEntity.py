from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.models import StrategyRequest
from typing import List, Optional


class StrategyRequestEntity(Entity):
    entity_name = EntityEnum.ENTITY

    def __init__(self):
        super().__init__()
        self.strategy_name = None
        self.strategy_path = None
        self.param_config = {}  # Initialize as empty dict instead of None
        self._nested_requests: List[StrategyRequestEntity] = []  # Store nested requests
        self.created_at = None
        self.updated_at = None
        self.id = None
        self.ret_val = {}
        self.is_applied = False  # Flag to indicate if the strategy has been applied
        self.add_to_history = True  # Flag to indicate if the strategy should be added to the history

    def add_nested_request(self, request: 'StrategyRequestEntity'):
        """Add a nested strategy request"""
        if not isinstance(request, StrategyRequestEntity):
            raise ValueError("Nested request must be a StrategyRequestEntity")
        self._nested_requests.append(request)

    def get_nested_requests(self) -> List['StrategyRequestEntity']:
        """Get all nested strategy requests"""
        return self._nested_requests

    def remove_nested_request(self, request: 'StrategyRequestEntity'):
        """Remove a nested strategy request"""
        if request in self._nested_requests:
            self._nested_requests.remove(request)

    def to_db(self):
        return StrategyRequestAdapter.entity_to_model(self)

    @classmethod
    def from_db(cls, data):
        return StrategyRequestAdapter.model_to_entity(data)

    def serialize(self):
        return {
            'name': self.strategy_name,
            'config': {
                'strategy_name': self.strategy_name,
                'param_config': self.param_config,
                'strategy_path': self.strategy_path,
            }
        }


class StrategyRequestAdapter:
    @staticmethod
    def model_to_entity(model: StrategyRequest) -> StrategyRequestEntity:
        """Convert a StrategyRequest model to a StrategyRequestEntity"""
        entity = StrategyRequestEntity()
        entity.id = model.pk
        entity.strategy_name = model.strategy_name
        entity.param_config = model.param_config
        entity.strategy_path = model.strategy_path
        entity.created_at = model.created_at
        entity.updated_at = model.updated_at
        entity.add_to_history = model.add_to_history

        # Convert nested requests
        for nested_request in model.nested_requests.all():
            entity.add_nested_request(StrategyRequestAdapter.model_to_entity(nested_request))

        return entity

    @staticmethod
    def entity_to_model(entity: StrategyRequestEntity, model: Optional[StrategyRequest] = None) -> StrategyRequest:
        """Convert a StrategyRequestEntity to a StrategyRequest model"""
        if model is None:
            if entity.id is not None:
                model = StrategyRequest.objects.get(id=entity.id)
            else:
                model = StrategyRequest()

        # Update model fields
        model.strategy_name = entity.strategy_name
        model.param_config = entity.param_config
        model.strategy_path = entity.strategy_path
        model.created_at = entity.created_at
        model.updated_at = entity.updated_at
        model.add_to_history = entity.add_to_history

        if not model.pk:  # If this is a new model
            model.save()

        # Handle nested requests
        for nested_request in entity.get_nested_requests():
            nested_model = StrategyRequestAdapter.entity_to_model(nested_request)
            model.nested_requests.add(nested_model)

        return model


