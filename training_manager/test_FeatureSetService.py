import numpy as np
from django.test import TestCase
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance

from training_manager.FeatureSetService import MeanVarianceScaler3D, FeatureSetService
from training_manager.models import FeatureSet


class FeatureSetServiceTestCase(TestCase):

    def test_create_feature_set(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2']
        }
        feature_set_service = FeatureSetService()
        feature_set = feature_set_service.create_feature_set(feature_set_config)
        self.assertEqual(type(feature_set.scaler), type(MeanVarianceScaler3D()))
        self.assertEqual(feature_set.feature_list, ['feature1', 'feature2'])

        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'TimeSeriesScalerMeanVariance'
            },
            'feature_list': ['feature1', 'feature2']
        }
        self.assertRaises(ValueError, feature_set_service.create_feature_set, feature_set_config)





