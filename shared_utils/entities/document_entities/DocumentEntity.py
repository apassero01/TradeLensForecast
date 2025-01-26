from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional, Dict, Any
import numpy as np

class DocumentEntity(Entity):
    entity_name = EntityEnum.DOCUMENT  # Update this line

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute('text', '')
        self.set_attribute('vector', None)
        self.set_attribute('document_type', None)
        self.set_attribute('processed_text', '')
        self.set_attribute('tokens', [])

    def set_text(self, text: str):
        """Set the document's raw text"""
        self.set_attribute('text', text)

    def get_text(self) -> str:
        """Get the document's raw text"""
        return self.get_attribute('text')

    def set_vector(self, vector: np.ndarray):
        """Set the document's vector representation"""
        self.set_attribute('vector', vector)

    def get_vector(self) -> Optional[np.ndarray]:
        """Get the document's vector representation"""
        return self.get_attribute('vector')

    def set_document_type(self, doc_type: str):
        """Set document type"""
        self.set_attribute('document_type', doc_type)

    def get_document_type(self) -> str:
        """Get document type"""
        return self.get_attribute('document_type')

    def set_processed_text(self, processed_text: str):
        """Set processed text"""
        self.set_attribute('processed_text', processed_text)

    def get_processed_text(self) -> str:
        """Get processed text"""
        return self.get_attribute('processed_text')

    def set_tokens(self, tokens: list):
        self.set_attribute('tokens', tokens)

    def get_tokens(self) -> list:
        return self.get_attribute('tokens')

    # def to_db(self):
    #     """Convert entity to database model using adapter"""
    #     from shared_utils.entities.document_entities.DocumentEntityAdapter import DocumentEntityAdapter
    #     return DocumentEntityAdapter.entity_to_model(self)
    #
    # @classmethod
    # def from_db(cls, data):
    #     """Create entity from database model using adapter"""
    #     from shared_utils.entities.document_entities.DocumentEntityAdapter import DocumentEntityAdapter
    #     return DocumentEntityAdapter.model_to_entity(data, cls)

    def serialize(self) -> dict:
        sup_dict = super().serialize()
        sup_dict['meta_data'] = {
            'path': self.get_attribute('path').split('/')[-1] if self.has_attribute('path') else None,
            'text': len(self.get_text()) if self.has_attribute('text') else 0,
            'vector_shape': self.get_vector().shape if self.has_attribute('vector') and self.get_vector() is not None else None,
            'document_type': self.get_document_type() if self.has_attribute('document_type') else None,
            'tokens_count': len(self.get_tokens()) if self.has_attribute('tokens') else 0,
        }
        return sup_dict
