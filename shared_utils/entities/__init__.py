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
    from .VisualizationEntity import VisualizationEntity
    from model_stage.entities.ModelStageEntity import ModelStageEntity
    from .document_entities.DocumentEntity import DocumentEntity
    from .input_entity.InputEntity import InputEntity
