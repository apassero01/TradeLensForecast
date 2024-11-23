from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum


class ModelStageEntity(Entity):
    entity_name = EntityEnum.MODEL_STAGE
    def __init__(self, model_stage):
        super().__init__()
        self.model_stage = model_stage

    @staticmethod
    def get_maximum_members():
        return {}



class ModelEntity(Entity):
    entity_name = EntityEnum.MODEL
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def get_maximum_members():
        return {}