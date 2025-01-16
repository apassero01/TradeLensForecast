from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from typing import Optional, Dict, Any
class FeatureSetEntity(Entity):
    entity_name = EntityEnum.FEATURE_SET
    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.feature_list = []
        self.scaler_config = {}
        self.do_fit_test = False
        self.secondary_feature_list = []
        self.feature_set_type = None

    def to_db(self):
        raise NotImplementedError("Child classes must implement the 'to_db' method.")

    @classmethod
    def from_db(cls, data):
        raise NotImplementedError("Child classes must implement the 'from_db' method.")