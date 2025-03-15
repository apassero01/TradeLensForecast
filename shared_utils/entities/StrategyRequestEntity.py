from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EntityModel import EntityModel
from shared_utils.models import StrategyRequest
from typing import List, Optional


class StrategyRequestEntity(Entity):
    entity_name = EntityEnum.STRATEGY_REQUEST

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.strategy_name = "None"
        self.param_config = {}  # Initialize as empty dict instead of None
        self._nested_requests: List[StrategyRequestEntity] = []  # Store nested requests
        self.created_at = None
        self.updated_at = None
        self.id = None
        self.ret_val = {}
        self.is_applied = False  # Flag to indicate if the strategy has been applied
        self.add_to_history = False  # Flag to indicate if the strategy should be added to the history
        self.target_entity_id = self.parent_ids[0] if self.parent_ids else None
        self.set_attribute('width', 700)
        self.set_attribute('height', 500)
    def add_nested_request(self, request: 'StrategyRequestEntity'):
        """Add a nested strategy request"""
        if not isinstance(request, StrategyRequestEntity):
            raise ValueError("Nested request must be a StrategyRequestEntity")

        nested_request_ids = [nested.entity_id for nested in self._nested_requests]
        if request.entity_id in nested_request_ids:
            old_request = self._nested_requests[nested_request_ids.index(request.entity_id)]
            self._nested_requests.remove(old_request)
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

        sup_dict = super().serialize()
        sup_dict.update({
            'strategy_name': self.strategy_name,
            'param_config': self.param_config,
            'nested_requests': [nested_request.serialize() for nested_request in self._nested_requests],
            'add_to_history': self.add_to_history,
            'entity_id': self.entity_id,
            'target_entity_id': self.parent_ids[0] if len(self.parent_ids) > 0 else self.target_entity_id
        })
        return sup_dict


class StrategyRequestAdapter:
    @staticmethod
    def model_to_entity(model: StrategyRequest) -> StrategyRequestEntity:
        """Convert a StrategyRequest model to a StrategyRequestEntity"""
        entity_id = model.entity_id
        entity = StrategyRequestEntity(str(entity_id))

        # Map basic attributes
        entity.strategy_name = model.strategy_name
        entity.param_config = model.param_config
        entity.created_at = model.created_at
        entity.updated_at = model.updated_at
        entity.add_to_history = model.add_to_history
        entity.target_entity_id = model.target_entity_id

        # Map parent request (if it exists)
        if model.parent_request:
            entity.parent_request = str(model.parent_request.entity_id)

        # Map training session (entity_model as a UUID)
        if model.entity_model:
            entity.entity_model = str(model.entity_model.entity_id)

        # Convert nested requests
        for nested_request in model.nested_requests.all():  # related_name='nested_requests'
            nested_entity = StrategyRequestAdapter.model_to_entity(nested_request)
            entity.add_nested_request(nested_entity)

        return entity

    @staticmethod
    def entity_to_model(entity: StrategyRequestEntity, model: Optional[StrategyRequest] = None) -> StrategyRequest:
        """Convert a StrategyRequestEntity to a StrategyRequest model"""
        if model is None:
            try:
                # Attempt to fetch the existing model
                model = StrategyRequest.objects.get(entity_id=entity.entity_id)
            except StrategyRequest.DoesNotExist:
                # Create a new model instance if it does not exist
                model = StrategyRequest(entity_id=entity.entity_id)

        # Map basic attributes
        model.strategy_name = entity.strategy_name
        model.param_config = entity.param_config
        model.add_to_history = entity.add_to_history
        model.target_entity_id = entity.target_entity_id

        if model.strategy_name == None:
            model.strategy_name = "None"

        # Map parent request (if it exists)
        if hasattr(entity, 'parent_request') and entity.parent_request:
            try:
                parent_request = StrategyRequest.objects.get(entity_id=entity.parent_request)
                model.parent_request = parent_request
            except StrategyRequest.DoesNotExist:
                raise ValueError(f"Parent request with ID {entity.parent_request.entity_id} does not exist.")

        # Map training session (entity_model as a ForeignKey)
        if hasattr(entity, 'entity_model') and entity.entity_model:
            try:
                entity_model = EntityModel.objects.get(entity_id=entity.entity_model)
                model.entity_model = entity_model
            except EntityModel.DoesNotExist:
                raise ValueError(f"EntityModel with ID {entity.entity_model} does not exist.")

        # Save the updated or newly created model
        model.save()

        # Handle nested requests
        existing_nested_request_ids = set(model.nested_requests.values_list('entity_id', flat=True))
        for nested_request in entity.get_nested_requests():
            nested_model = StrategyRequestAdapter.entity_to_model(nested_request)
            if nested_model.entity_id not in existing_nested_request_ids:
                nested_model.parent_request = model  # Set the parent for the nested request
                nested_model.save()

        return model

