from typing import Optional
from uuid import uuid4
from datetime import datetime

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy


class CalendarEventEntity(Entity):
    entity_name = EntityEnum.CALENDAR_EVENT
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        # Initialize basic calendar event attributes
        created_date = datetime.now().date().isoformat()
        
        # Set attributes
        self.set_attribute('title', '')
        self.set_attribute('start_time', '')
        self.set_attribute('end_time', '')
        self.set_attribute('description', '')
        self.set_attribute('location', '')
        self.set_attribute('date', created_date) 
        self.set_attribute('hidden', False)

    def on_create(self, param_config: Optional[dict] = None) -> list:
        """Override this method to handle entity creation logic"""
        requests = []

        # Create event details view
        event_details_child_id = str(uuid4())
        event_details_view_attributes = {
            'parent_attributes': {
                "title": "title",
                "start_time": "start_time",
                "end_time": "end_time",
                "description": "description",
                "location": "location",
                "date": "date",
            },
            'view_component_type': 'calendar_event_details',
        }

        event_details_view_request = CreateEntityStrategy.request_constructor(
            self.entity_id,
            ViewEntity.get_class_path(),
            entity_uuid=event_details_child_id,
            initial_attributes=event_details_view_attributes
        )
        requests.append(event_details_view_request)

        return requests

    def serialize(self):
        """Override this method to handle entity serialization logic"""
        serialized_data = super().serialize()
        serialized_data['title'] = self.get_attribute('title')
        serialized_data['start_time'] = self.get_attribute('start_time')
        serialized_data['end_time'] = self.get_attribute('end_time')
        serialized_data['description'] = self.get_attribute('description')
        serialized_data['location'] = self.get_attribute('location')
        serialized_data['date'] = self.get_attribute('date')
        return serialized_data
