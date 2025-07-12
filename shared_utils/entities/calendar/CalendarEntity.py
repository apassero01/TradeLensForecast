from typing import Optional
from uuid import uuid4

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy


class CalendarEntity(Entity):
    entity_name = EntityEnum.CALENDAR
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        # Initialize basic calendar event attributes
        self.set_attribute('title', '')
        self.set_attribute("hidden", False)

    def on_create(self, param_config: Optional[dict] = None) -> list:
        """Override this method to handle entity creation logic"""
        requests = []

        # Create monthly events view
        monthly_child_id = str(uuid4())
        monthly_view_attributes = {
            'parent_attributes': {
                "title": "title",
            },
            'view_component_type': 'calendarmonthlyview',
        }
        monthly_view_request = CreateEntityStrategy.request_constructor(
            self.entity_id,
            ViewEntity.get_class_path(),
            entity_uuid=monthly_child_id,
            initial_attributes=monthly_view_attributes
        )
        requests.append(monthly_view_request)

        # Create weekly events view
        weekly_child_id = str(uuid4())
        weekly_view_attributes = {
            'parent_attributes': {
                "title": "title",
            },
            'view_component_type': 'calendarweeklyview',
        }
        weekly_view_request = CreateEntityStrategy.request_constructor(
            self.entity_id,
            ViewEntity.get_class_path(),
            entity_uuid=weekly_child_id,
            initial_attributes=weekly_view_attributes
        )
        requests.append(weekly_view_request)

        # Create yearly events view
        yearly_child_id = str(uuid4())
        yearly_view_attributes = {
            'parent_attributes': {
                "title": "title",
            },
            'view_component_type': 'calendaryearlyview',
        }
        yearly_view_request = CreateEntityStrategy.request_constructor(
            self.entity_id,
            ViewEntity.get_class_path(),
            entity_uuid=yearly_child_id,
            initial_attributes=yearly_view_attributes
        )
        requests.append(yearly_view_request)

        return requests

    def serialize(self):
        """Override this method to handle entity serialization logic"""
        serialized_data = super().serialize()
        serialized_data['title'] = self.get_attribute('title')
        return serialized_data
