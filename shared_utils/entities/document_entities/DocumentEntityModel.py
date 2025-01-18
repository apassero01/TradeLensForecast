from django.db import models
from pgvector.django import VectorField
from shared_utils.entities.EntityModel import EntityModel

class DocumentEntityModel(EntityModel):
    """
    Document entity model with vector storage support.
    Inherits core functionality from EntityModel.
    """
    # Vector field for efficient similarity searches
    vector = VectorField(dimensions=1536, null=True)  # 1536 for OpenAI embeddings
    
    class Meta:
        db_table = 'document_entities'
        indexes = [
            # Vector similarity search index
            models.Index(name='document_vector_idx', fields=['vector'], 
                        opclasses=['vector_cosine_ops'])
        ]
