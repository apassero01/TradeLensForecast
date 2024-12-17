from django.db import models
from django.core.exceptions import ValidationError
from training_session.models import TrainingSession
class StrategyRequest(models.Model):
    strategy_name = models.CharField(max_length=255)  # The name of the strategy
    strategy_path = models.CharField(max_length=255)  # The path to the strategy
    param_config = models.JSONField(default=dict)  # Parameters for the strategy
    
    # Link to the TrainingSession to specify which strategy history this request belongs to
    training_session = models.ForeignKey(
        TrainingSession, 
        on_delete=models.CASCADE, 
        related_name='strategy_requests',
        null=True,
        blank=True
    )  

    # Link for parent-child relationships
    parent_request = models.ForeignKey(
        'self', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True, 
        related_name='nested_requests'
    )  

    add_to_history = models.BooleanField(default=True)  # Should this be in the top-level strategy history?

    created_at = models.DateTimeField(auto_now_add=True)  # When the request was created
    updated_at = models.DateTimeField(auto_now=True)      # When the request was last updated
    
    entity_id = models.CharField(max_length=255, null=True, blank=True)  # Optional identifier for tracking

    def clean(self):
        """Enforce validation logic to prevent circular references and conflicting roles."""
        # Ensure it is not its own parent
        if self.pk and self.parent_request_id == self.pk:
            raise ValidationError("A StrategyRequest cannot be its own parent.")
        
        # If it has a parent, it shouldn't be part of the top-level strategy history
        if self.parent_request is not None and self.add_to_history:
            raise ValidationError("A nested StrategyRequest cannot be part of the strategy history.")
    
    def is_top_level(self):
        """Check if this request is a top-level request (has no parent)"""
        return self.parent_request is None

    def __str__(self):
        return f"StrategyRequest(name={self.strategy_name}, parent={self.parent_request_id}, training_session={self.training_session_id})"





