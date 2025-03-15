from django.db import models
from django.core.exceptions import ValidationError
from shared_utils.entities.EntityModel import EntityModel
import uuid
class StrategyRequest(EntityModel):
    """
    A request for a strategy to be applied to an entity.
    """
    strategy_name = models.CharField(max_length=255)  # The name of the strategy
    param_config = models.JSONField(default=dict)  # Parameters for the strategy
    target_entity_id = models.CharField(max_length=255, null=True, blank=True)  # The id of the target entity
    
    # Link to the EntityModel to specify which entity this request belongs to
    entity_model = models.ForeignKey(
        EntityModel, 
        on_delete=models.CASCADE, 
        related_name='strategy_requests',
        null=True,
        blank=True,
        to_field='entity_id'
    )  

    # Link for parent-child relationships
    parent_request = models.ForeignKey(
        'self', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True, 
        related_name='nested_requests',
        to_field='entity_id'
    )  

    add_to_history = models.BooleanField(default=True)  # Should this be in the top-level strategy history?

    created_at = models.DateTimeField(auto_now_add=True)  # When the request was created
    updated_at = models.DateTimeField(auto_now=True)      # When the request was last updated
    
    def clean(self):
        """Enforce validation logic to prevent circular references and conflicting roles."""
        # Ensure it is not its own parent
        if self.entity_id and self.parent_request_id == self.entity_id:
            raise ValidationError("A StrategyRequest cannot be its own parent.")
        
        # If it has a parent, it shouldn't be part of the top-level strategy history
        if self.parent_request is not None and self.add_to_history:
            raise ValidationError("A nested StrategyRequest cannot be part of the strategy history.")
    
    def is_top_level(self):
        """Check if this request is a top-level request (has no parent)"""
        return self.parent_request is None

    def __str__(self):
        return f"StrategyRequest(name={self.strategy_name}, parent={self.parent_request_id}, entity={self.entity_model.entity_id if self.entity_model else None})"





