from enum import Enum

import numpy as np

from data_bundle_manager.entities.FeatureSetEntity import FeatureSetEntity
from data_bundle_manager.scalers.scalers import ScalerEnum


class FeatureSetEntityService:
    def create_feature_set(self, feature_set_config):
        scaler_config = feature_set_config['scaler_config']
        scaler_name = scaler_config['scaler_name']
        scaler = self.get_scaler(scaler_name)
        do_fit_test = feature_set_config['do_fit_test']
        feature_set_type = feature_set_config['feature_set_type']

        feature_list = feature_set_config['feature_list']

        feature_set = FeatureSetEntity()
        feature_set.feature_list = feature_list
        feature_set.do_fit_test = do_fit_test
        feature_set.feature_set_type = feature_set_type
        feature_set.secondary_feature_list = feature_set_config.get('secondary_feature_list', None)
        feature_set.scaler = scaler

        return feature_set

    def get_scaler(self, scaler_name, scaler_config = None):

        if scaler_name not in ScalerEnum.__members__:
            raise ValueError(f"Scaler {scaler_name} not supported")

        scaler_class = ScalerEnum[scaler_name].value
        if scaler_config:
            scaler = scaler_class()
            scaler.deserialize(scaler_config)
            return scaler
        else:
            return scaler_class()

