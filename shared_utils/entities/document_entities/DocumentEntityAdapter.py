from typing import Optional, Type
import numpy as np

from shared_utils.entities.Entity import EntityAdapter
from shared_utils.entities.document_entities.DocumentEntity import DocumentEntity
from shared_utils.entities.document_entities.DocumentEntityModel import DocumentEntityModel

class DocumentEntityAdapter(EntityAdapter):
    """
    Adapter for converting between DocumentEntity and DocumentEntityModel.
    Handles special case of vector storage.
    """
    
    @classmethod
    def model_to_entity(cls, model: DocumentEntityModel, entity_class: Type[DocumentEntity] = DocumentEntity) -> DocumentEntity:
        """Convert a DocumentEntityModel to a DocumentEntity"""
        # First use parent class to handle basic conversion
        entity = super().model_to_entity(model, entity_class)
        
        # If vector exists in model, add it to attributes
        if model.vector is not None:
            entity.set_vector(np.array(model.vector))
            
        return entity

    @classmethod
    def entity_to_model(cls, entity: DocumentEntity, 
                       model: Optional[DocumentEntityModel] = None, 
                       model_class: Type[DocumentEntityModel] = DocumentEntityModel) -> DocumentEntityModel:
        """Convert a DocumentEntity to a DocumentEntityModel"""
        if model is None:
            model = super().entity_to_model(entity, model_class=model_class)
        else:
            model = super().entity_to_model(entity, model=model)
            
        # Handle vector field
        if entity.has_attribute('vector'):
            # Extract vector from attributes and store in dedicated column
            vector = entity.get_vector()
            if vector is not None:
                model.vector = vector.tolist()  # Convert numpy array to list for storage
                
            # Remove vector from attributes to avoid duplicate storage
            attributes = model.attributes.copy()
            attributes.pop('vector', None)
            model.attributes = attributes
            
        return model
