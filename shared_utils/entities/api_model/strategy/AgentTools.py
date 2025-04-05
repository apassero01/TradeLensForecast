from langchain_core.tools import tool
from typing import List


@tool
def serialize_entities(self, entities: List[str]) -> List[dict]:
    '''
    Serialize a list of entities. If the model needs to know about entities with
    specific ids, this method will return more information about the entities.
    '''
    serialized_entities = []
    for entity_id in entities:
        entity = self.entity_service.get_entity(entity_id)
        if entity:
            serialized_entities.append(entity.serialize())
    return serialized_entities