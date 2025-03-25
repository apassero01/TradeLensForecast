from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from typing import Optional, Dict, Any

class DataBundleEntity(Entity):
    entity_name = EntityEnum.DATA_BUNDLE

    def __init__(self, entity_id: Optional[str] = None):
        super().__init__(entity_id)
        self.set_attribute('width', 300)
        self.set_attribute('height', 500)

    def on_create(self, param_config: Optional[Dict[str, Any]] = None):
        pass

    def serialize(self):
        sup_dict = super().serialize()
        sup_dict['meta_data'] =  {
            'X': self.get_attribute('X').shape if self.has_attribute('X') else None,
            'y': self.get_attribute('y').shape if self.has_attribute('y') else None,
            'X_train': self.get_attribute('X_train').shape if self.has_attribute('X_train') else None,
            'X_test': self.get_attribute('X_test').shape if self.has_attribute('X_test') else None,
            'y_train': self.get_attribute('y_train').shape if self.has_attribute('y_train') else None,
            'y_test': self.get_attribute('y_test').shape if self.has_attribute('y_test') else None,
            'y_train_scaled': self.get_attribute('y_train_scaled').shape if self.has_attribute(
                'y_train_scaled') else None,
            'y_test_scaled': self.get_attribute('y_test_scaled').shape if self.has_attribute('y_test_scaled') else None,
            'X_train_scaled': self.get_attribute('X_train_scaled').shape if self.has_attribute(
                'X_train_scaled') else None,
            'X_test_scaled': self.get_attribute('X_test_scaled').shape if self.has_attribute('X_test_scaled') else None,
        }

        return sup_dict

