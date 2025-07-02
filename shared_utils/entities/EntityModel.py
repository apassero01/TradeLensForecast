from django.db import models
from django.db.models import JSONField
import uuid
from pgvector.django import VectorField
from pgvector.django import HnswIndex


class EntityModel(models.Model):
    """
    Base model for all entities in the system.
    Stores core entity data and provides common functionality.
    """
    entity_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    entity_type = models.CharField(max_length=50)  # From EntityEnum
    attributes = JSONField(default=dict, db_column='attributes')
    children_ids = JSONField(default=list)
    parent_ids = JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class_path = models.CharField(max_length=255)  # Full path to entity class

    embedding = VectorField(dimensions=384, null=True)

    class Meta:
        indexes = [
            HnswIndex(
                name='entity_embedding_hnsw',
                fields=['embedding'],
                opclasses=['vector_cosine_ops'],   # or vector_l2_ops / vector_ip_ops
                m=16,               # optional HNSW knobs (default 16 / 64)
                ef_construction=64
            )
        ]
