
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity


class DataBundleEntity(Entity):
    entity_name = EntityEnum.DATA_BUNDLE

    def __init__(self):
        super().__init__()
        self.dataset = {}

    def to_db(self):
        raise NotImplementedError("Child classes must implement the 'to_db' method.")
    @classmethod
    def from_db(cls, data):
        raise NotImplementedError("Child classes must implement the 'from_db' method.")

    def set_dataset(self, dataset):
        for key, value in dataset.items():
            self.dataset[key] = value

    @staticmethod
    def get_maximum_members():
        return {
            EntityEnum.FEATURE_SETS: None,
        }

    def serialize(self):
        serialized_children = []
        for key, value in self.entity_map.items():
            if isinstance(value, Entity):
                serialized_children.append(value.serialize())
            else:
                for v in value:
                    serialized_children.append(v.serialize())

        return {
            'entity_name': self.entity_name.value,
            'children': serialized_children,
            'meta_data': {
                'X' : self.dataset['X'].shape if 'X' in self.dataset else None,
                'y' : self.dataset['y'].shape if 'y' in self.dataset else None,
                'X_train' : self.dataset['X_train'].shape if 'X_train' in self.dataset else None,
                'X_test' : self.dataset['X_test'].shape if 'X_test' in self.dataset else None,
                'y_train' : self.dataset['y_train'].shape if 'y_train' in self.dataset else None,
                'y_test' : self.dataset['y_test'].shape if 'y_test' in self.dataset else None,
                'y_train_scaled' : self.dataset['y_train_scaled'].shape if 'y_train_scaled' in self.dataset else None,
                'y_test_scaled' : self.dataset['y_test_scaled'].shape if 'y_test_scaled' in self.dataset else None,
                'X_train_scaled' : self.dataset['X_train_scaled'].shape if 'X_train_scaled' in self.dataset else None,
                'X_test_scaled' : self.dataset['X_test_scaled'].shape if 'X_test_scaled' in self.dataset else None,

            }
        }
