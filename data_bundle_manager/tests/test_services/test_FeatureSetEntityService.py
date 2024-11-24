from django.test import TestCase
import numpy as np
from data_bundle_manager.entities.FeatureSetEntity import FeatureSetEntity
from data_bundle_manager.entities.services.FeatureSetEntityService import FeatureSetEntityService
from data_bundle_manager.scalers.scalers import MeanVarianceScaler3D


class FeatureSetEntityServiceTestCase(TestCase):

    def test_create_feature_set(self):
        feature_set_config = {
            'scaler_config': {
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'do_fit_test': False,
            'feature_set_type': 'X'
        }
        feature_set_service = FeatureSetEntityService()
        feature_set = feature_set_service.create_feature_set(feature_set_config)

        # Check if the feature set is an instance of FeatureSetEntity
        self.assertIsInstance(feature_set, FeatureSetEntity)

        # Check attributes
        self.assertEqual(type(feature_set.scaler), MeanVarianceScaler3D)
        self.assertEqual(feature_set.feature_list, ['feature1', 'feature2'])
        self.assertFalse(feature_set.do_fit_test)
        self.assertEqual(feature_set.feature_set_type, 'X')

    def test_create_feature_set_with_secondary_features(self):
        feature_set_config = {
            'scaler_config': {
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'secondary_feature_list': ['feature3', 'feature4'],
            'feature_set_type': 'Y',
            'do_fit_test': True
        }
        feature_set_service = FeatureSetEntityService()
        feature_set = feature_set_service.create_feature_set(feature_set_config)

        # Check attributes
        self.assertEqual(type(feature_set.scaler), MeanVarianceScaler3D)
        self.assertEqual(feature_set.feature_list, ['feature1', 'feature2'])
        self.assertEqual(feature_set.secondary_feature_list, ['feature3', 'feature4'])
        self.assertTrue(feature_set.do_fit_test)
        self.assertEqual(feature_set.feature_set_type, 'Y')

    def test_get_scaler(self):
        feature_set_service = FeatureSetEntityService()
        scaler = feature_set_service.get_scaler('MEAN_VARIANCE_SCALER_3D')
        self.assertIsInstance(scaler, MeanVarianceScaler3D)

        scaler_params = {
            'mean': [1, 2],
            'var': [3, 4]
        }

        scaler = feature_set_service.get_scaler('MEAN_VARIANCE_SCALER_3D', scaler_params)
        self.assertIsInstance(scaler, MeanVarianceScaler3D)
        np.testing.assert_almost_equal(scaler.mean_, [1, 2])
        np.testing.assert_almost_equal(scaler.var_, [3, 4])