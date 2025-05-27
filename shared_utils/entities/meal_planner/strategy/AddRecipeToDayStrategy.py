from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum


class AddRecipeToDayStrategy(Strategy):
    """
    Strategy for adding a recipe ID to a specific day of the week in a MealPlanEntity.
    """

    entity_type = EntityEnum.MEAL_PLAN
    strategy_description = 'Adds a recipe ID to a specific day of the week in a meal plan'

    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        """
        Validate we have the required parameters: day and recipe_id
        """
        config = strategy_request.param_config
        
        if 'day' not in config:
            raise ValueError("param_config must include 'day' (e.g., 'monday', 'tuesday', etc.)")
        
        if 'recipe_id' not in config:
            raise ValueError("param_config must include 'recipe_id' to add to the day")
        
        # Validate day is valid
        day = config['day'].lower()
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if day not in valid_days:
            raise ValueError(f"Invalid day: {day}. Must be one of {valid_days}")

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Add a recipe ID to the specified day of the week in the meal plan.
        """
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config
        
        day = config.get('day').lower()
        recipe_id = config.get('recipe_id')
        
        # Use the entity's method to add the recipe
        entity.add_recipe_to_day(day, recipe_id)
        
        # Save the entity
        self.entity_service.save_entity(entity)
        
        # Store result information
        self.strategy_request.ret_val['day'] = day
        self.strategy_request.ret_val['recipe_id'] = recipe_id
        self.strategy_request.ret_val['success'] = True
        
        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Default config for AddRecipeToDayStrategy.
        """
        return {
            "strategy_name": "AddRecipeToDayStrategy",
            "strategy_path": None,
            "param_config": {
                "day": "monday",
                "recipe_id": ""
            }
        }

    @classmethod
    def request_constructor(cls, target_entity_id: str, day: str, recipe_id: str, add_to_history: bool = False):
        """
        Convenience method to construct a strategy request for adding a recipe to a day.
        """
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = cls.__name__
        strategy_request.param_config = {
            "day": day,
            "recipe_id": recipe_id
        }
        strategy_request.target_entity_id = target_entity_id
        strategy_request.add_to_history = add_to_history
        strategy_request._nested_requests = []
        return strategy_request 