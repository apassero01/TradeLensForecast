from uuid import uuid4

from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional, Dict, Any
import numpy as np

from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.VisualizationEntity import VisualizationEntity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.models import StrategyRequest
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy, SetAttributesStrategy
from shared_utils.strategy.VisualizationStrategy import VisualizationStrategy


class DocumentEntity(Entity):
    entity_name = EntityEnum.DOCUMENT  # Update this line

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute('text', '')
        self.set_attribute('vector', None)
        self.set_attribute('document_type', None)
        self.set_attribute('processed_text', '')
        self.set_attribute('tokens', [])

    def on_create(self, param_config: Optional[Dict[str, Any]] = None) -> list[StrategyRequestEntity]:
        """Override this method to handle entity creation logic"""
        strategy_request_list = []
        child_vis_create  = StrategyRequestEntity()
        self.add_child(child_vis_create)

        child_vis_create.strategy_name = CreateEntityStrategy.__name__
        child_uuid = str(uuid4())
        child_vis_create.param_config = {
            'entity_class': ViewEntity.get_class_path(),
            'entity_uuid': child_uuid,
        }
        child_vis_create.target_entity_id = self.entity_id
        child_vis_create.add_to_history = False

        strategy_request_list.append(child_vis_create)

        child_vis_viz  = StrategyRequestEntity()
        self.add_child(child_vis_viz)
        child_vis_viz.strategy_name = SetAttributesStrategy.__name__
        child_vis_viz.param_config['attribute_map'] = {
            'parent_attributes': {"text":"text"},
            'view_component_type': 'editor',
        }

        child_vis_viz.target_entity_id = child_uuid
        child_vis_viz.add_to_history = False

        strategy_request_list.append(child_vis_viz)



        return strategy_request_list




        # Add the child request to the parent entity



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
        sup_dict['text'] = self.get_text()
        sup_dict['document_type'] = self.get_document_type() if self.has_attribute('document_type') else None
        return sup_dict
