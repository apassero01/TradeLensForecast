from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional

class ViewEntity(Entity):
    entity_name = EntityEnum.VIEW
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute("parent_attributes", {})
        self.set_attribute("view_component_type", None)
        self.set_attribute("view_component", None)

    def serialize(self) -> dict:
        parent_dict = super().serialize()
        parent_dict["parent_attributes"] = self.get_attribute("parent_attributes")
        parent_dict["view_component_type"] = self.get_attribute("view_component_type")
        parent_dict["view_component"] = self.get_attribute("view_component")
        return parent_dict
