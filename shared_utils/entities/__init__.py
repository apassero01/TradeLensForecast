# shared_utils/entities/__init__.py

# Import all entity classes here so they're loaded when Django starts
# from .Entity import Entity
# # from .StrategyRequestEntity import StrategyRequestEntity
# from .VisualizationEntity import VisualizationEntity
# from model_stage.entities.ModelStageEntity import ModelStageEntity
# from .document_entities.DocumentEntity import DocumentEntity
# from .api_model.ApiModelEntity import ApiModelEntity

def discover_entities():
    from .Entity import Entity
    from .StrategyRequestEntity import StrategyRequestEntity 
    from .api_model.ApiModelEntity import ApiModelEntity
    from model_stage.entities.ModelStageEntity import ModelStageEntity
    from .document_entities.DocumentEntity import DocumentEntity
    from .view_entity.ViewEntity import ViewEntity
    from .meal_planner.RecipeEntity import RecipeEntity
    from .meal_planner.MealPlanEntity import MealPlanEntity
    from .calendar.CalendarEventEntity import CalendarEventEntity
    from .calendar.CalendarEntity import CalendarEntity

