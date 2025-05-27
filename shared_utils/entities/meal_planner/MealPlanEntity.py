from typing import Optional
from uuid import uuid4

from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy


class MealPlanEntity(Entity):
    entity_name = EntityEnum.MEAL_PLAN
    
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        
        # Initialize attributes for each day of the week with empty lists
        self.set_attribute('monday', [])
        self.set_attribute('tuesday', [])
        self.set_attribute('wednesday', [])
        self.set_attribute('thursday', [])
        self.set_attribute('friday', [])
        self.set_attribute('saturday', [])
        self.set_attribute('sunday', [])
        
        # Additional meal plan attributes
        self.set_attribute('name', '')
        self.set_attribute('week_start_date', '')
        self.set_attribute("hidden", True)

    def on_create(self, param_config: Optional[dict] = None) -> list:
        """Override this method to handle entity creation logic should return list of requests to operate"""
        requests = []
        
        # Create a view for the meal plan
        meal_plan_view_child_id = str(uuid4())
        meal_plan_view_attributes = {
            'parent_attributes': {
                "name": "name",
                "week_start_date": "week_start_date",
                "monday": "monday",
                "tuesday": "tuesday", 
                "wednesday": "wednesday",
                "thursday": "thursday",
                "friday": "friday",
                "saturday": "saturday",
                "sunday": "sunday",
                "hidden": "false"
            },
            'view_component_type': 'mealplan',
        }
        meal_plan_view_request = CreateEntityStrategy.request_constructor(
            self.entity_id, 
            ViewEntity.get_class_path(), 
            entity_uuid=meal_plan_view_child_id, 
            initial_attributes=meal_plan_view_attributes
        )
        requests.append(meal_plan_view_request)

        return requests

    def serialize(self):
        """Override this method to handle entity serialization logic"""
        serialized_data = super().serialize()
        serialized_data['name'] = self.get_attribute('name')
        serialized_data['week_start_date'] = self.get_attribute('week_start_date')
        serialized_data['monday'] = self.get_attribute('monday')
        serialized_data['tuesday'] = self.get_attribute('tuesday')
        serialized_data['wednesday'] = self.get_attribute('wednesday')
        serialized_data['thursday'] = self.get_attribute('thursday')
        serialized_data['friday'] = self.get_attribute('friday')
        serialized_data['saturday'] = self.get_attribute('saturday')
        serialized_data['sunday'] = self.get_attribute('sunday')
        return serialized_data

    def add_recipe_to_day(self, day: str, recipe_id: str):
        """Add a recipe ID to a specific day of the week"""
        day = day.lower()
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if day not in valid_days:
            raise ValueError(f"Invalid day: {day}. Must be one of {valid_days}")
        
        current_recipes = self.get_attribute(day)
        if recipe_id not in current_recipes:
            current_recipes.append(recipe_id)
            self.set_attribute(day, current_recipes)

    def remove_recipe_from_day(self, day: str, recipe_id: str):
        """Remove a recipe ID from a specific day of the week"""
        day = day.lower()
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if day not in valid_days:
            raise ValueError(f"Invalid day: {day}. Must be one of {valid_days}")
        
        current_recipes = self.get_attribute(day)
        if recipe_id in current_recipes:
            current_recipes.remove(recipe_id)
            self.set_attribute(day, current_recipes)

    def get_recipes_for_day(self, day: str) -> list:
        """Get all recipe IDs for a specific day of the week"""
        day = day.lower()
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        if day not in valid_days:
            raise ValueError(f"Invalid day: {day}. Must be one of {valid_days}")
        
        return self.get_attribute(day)

    def get_all_recipe_ids(self) -> list:
        """Get all unique recipe IDs across all days of the week"""
        all_recipe_ids = set()
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        
        for day in days:
            recipes = self.get_attribute(day)
            all_recipe_ids.update(recipes)
        
        return list(all_recipe_ids)

    @classmethod
    def get_class_path(cls):
        """Return the full class path for this entity"""
        return f"{cls.__module__}.{cls.__name__}" 