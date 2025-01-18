from django.db import models
from django.contrib.postgres.fields import JSONField
import uuid

class EntityModel(models.Model):
    """
    Base model for all entities in the system.
    Stores core entity data and provides common functionality.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entity_type = models.CharField(max_length=50)  # From EntityEnum
    attributes = JSONField(default=dict, db_column='attributes')
    children_ids = JSONField(default=list)
    parent_ids = JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class_path = models.CharField(max_length=255)  # Full path to entity class
    
    class Meta:
        abstract = True 