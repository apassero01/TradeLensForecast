from django.db import models

class StrategyRequest(models.Model):

    strategy_name = models.CharField(max_length=255)  # The name of the strategy
    strategy_path = models.CharField(max_length=255)  # The path to the strategy
    param_config = models.JSONField(default=dict)  # Parameters for the strategy
    nested_requests = models.ManyToManyField('self', symmetrical=False, related_name='parent_requests')  # Nested

    created_at = models.DateTimeField(auto_now_add=True)  # Automatically track when the request is created
    updated_at = models.DateTimeField(auto_now=True)      # Automatically track when the request is updated





