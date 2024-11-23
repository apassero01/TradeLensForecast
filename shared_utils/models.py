from django.db import models

class StrategyRequest(models.Model):

    strategy_name = models.CharField(max_length=255)  # The name of the strategy
    strategy_path = models.CharField(max_length=255)  # The path to the strategy
    param_config = models.JSONField(default=dict)  # Parameters for the strategy
    nested_requests = models.ManyToManyField('self', symmetrical=False, related_name='parent_requests')  # Nested

    created_at = models.DateTimeField(auto_now_add=True)  # Automatically track when the request is created
    updated_at = models.DateTimeField(auto_now=True)      # Automatically track when the request is updated

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an empty in-memory list for nested requests
        self.nested_requests_list = []

    def save(self, *args, **kwargs):
        # Populate the in-memory list only after saving the object
        super().save(*args, **kwargs)  # Call the parent's save method
        if not self.nested_requests_list:
            self.nested_requests_list = list(self.nested_requests.all())

    def add_nested_request(self, strategy_request):
        """Add a strategy request to the in-memory list."""
        if strategy_request not in self.nested_requests_list:
            self.nested_requests_list.append(strategy_request)

    def get_nested_requests(self):
        """Retrieve all nested requests (in-memory)."""
        return self.nested_requests_list


