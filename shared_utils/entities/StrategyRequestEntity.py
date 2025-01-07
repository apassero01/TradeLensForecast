from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.models import StrategyRequest
from typing import List, Optional


class StrategyRequestEntity(Entity):
    entity_name = EntityEnum.ENTITY

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
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
        self.target_entity_id = None
    def add_nested_request(self, request: 'StrategyRequestEntity'):
        """Add a nested strategy request"""
        if not isinstance(request, StrategyRequestEntity):
            raise ValueError("Nested request must be a StrategyRequestEntity")
        self._nested_requests.append(request)

    def add_nested_requests(self, requests: List['StrategyRequestEntity']):
        """Add a list of nested strategy requests"""
        for request in requests:
            self.add_nested_request(request)

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
            'strategy_name': self.strategy_name,
            'strategy_path': self.strategy_path,
            'param_config': self.param_config,
            'nested_requests': [nested_request.serialize() for nested_request in self._nested_requests],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'add_to_history': self.add_to_history,
            'entity_id': self.entity_id,
            'target_entity_id': self.target_entity_id
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
        
        # Handle the parent request (only if it exists)
        if model.parent_request:
            entity.parent_request_id = model.parent_request.pk
        
        # Handle training session (only if it exists)
        if model.training_session:
            entity.training_session_id = model.training_session.pk

        # Convert nested requests using the ForeignKey relationship
        for nested_request in model.nested_requests.all():  # ForeignKey related_name='nested_requests'
            nested_entity = StrategyRequestAdapter.model_to_entity(nested_request)
            entity.add_nested_request(nested_entity)

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
        model.add_to_history = entity.add_to_history

        # Handle parent request (if parent exists)
        if hasattr(entity, 'parent_request_id') and entity.parent_request_id:
            model.parent_request_id = entity.parent_request_id

        # Handle training session (if it exists)
        if hasattr(entity, 'training_session_id') and entity.training_session_id:
            model.training_session_id = entity.training_session_id

        if not model.pk:  # If this is a new model
            model.save()

        # Handle nested requests
        existing_nested_request_ids = set(model.nested_requests.values_list('id', flat=True))
        for nested_request in entity.get_nested_requests():
            nested_model = StrategyRequestAdapter.entity_to_model(nested_request)
            if nested_model.pk not in existing_nested_request_ids:
                nested_model.parent_request = model  # Set the parent for the nested request
                nested_model.save()

        return model


