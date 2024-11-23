from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity


class TrainingSessionEntity(Entity):
    entity_name = EntityEnum.TRAINING_SESSION
    def __init__(self, session):
        super().__init__()
        self.session = session

    @staticmethod
    def get_maximum_members():
        return{
            EntityEnum.MODEL_STAGE: None,
        }



