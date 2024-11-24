from shared_utils.entities.Entity import Entity

class FeatureSetEntity(Entity):
    def __init__(self):
        super().__init__()
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