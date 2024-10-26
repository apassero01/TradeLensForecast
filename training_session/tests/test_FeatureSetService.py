import json

import numpy as np
from django.test import TestCase
from training_session.services.FeatureSetService import FeatureSetService, MeanVarianceScaler3D

class FeatureSetServiceTestCase(TestCase):

    def test_create_feature_set(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'do_fit_test' : False
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

    def test_create_feature_set_with_y(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'secondary_feature_list': ['feature3', 'feature4'],
            'do_fit_test' : False
        }
        feature_set_service = FeatureSetService()
        feature_set = feature_set_service.create_feature_set(feature_set_config)
        self.assertEqual(type(feature_set.scaler), type(MeanVarianceScaler3D()))
        self.assertEqual(feature_set.feature_list, ['feature1', 'feature2'])
        self.assertEqual(feature_set.secondary_feature_list, ['feature3', 'feature4'])

    def test_get_scaler(self):
        feature_set_service = FeatureSetService()
        scaler = feature_set_service.get_scaler('MEAN_VARIANCE_SCALER_3D')
        self.assertEqual(type(scaler), type(MeanVarianceScaler3D()))

        scaler_params = {
            'mean': [1, 2],
            'var': [3, 4]
        }

        scaler = feature_set_service.get_scaler('MEAN_VARIANCE_SCALER_3D', scaler_params)
        self.assertEqual(type(scaler), type(MeanVarianceScaler3D()))
        np.testing.assert_almost_equal(scaler.mean_, [1, 2])
        np.testing.assert_almost_equal(scaler.var_, [3, 4])


class MeanVarianceScaler3DTestCase(TestCase):
    def test_fit(self):
        scaler = MeanVarianceScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        self.assertEqual(scaler.mean_.shape, (2,))
        self.assertEqual(scaler.var_.shape, (2,))

    def test_transform(self):
        scaler = MeanVarianceScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, (10, 5, 2))

    def test_inverse_transform(self):
        scaler = MeanVarianceScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, (10, 5, 2))
        np.testing.assert_almost_equal(X, X_inverse)

    def test_serialize(self):
        scaler = MeanVarianceScaler3D()
        config = scaler.serialize()
        self.assertEqual(config, {})

        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()
        self.assertEqual(len(config['mean']), 2)
        self.assertEqual(len(config['var']), 2)

        try:
            json.dumps(config)
        except Exception as e:
            self.fail("json.dump raised an exception unexpectedly! " + str(e))

    def test_deserialize(self):
        scaler = MeanVarianceScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()

        scaler2 = MeanVarianceScaler3D()
        scaler2.deserialize(config)
        np.testing.assert_almost_equal(scaler.mean_, scaler2.mean_)
        np.testing.assert_almost_equal(scaler.var_, scaler2.var_)

        np.testing.assert_almost_equal(scaler2.inverse_transform(scaler.fit_transform(X)), X)