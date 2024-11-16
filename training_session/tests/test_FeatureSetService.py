import json

import numpy as np
from django.test import TestCase
from training_session.services.FeatureSetService import FeatureSetService, MeanVarianceScaler3D, MinMaxSeqBySeqScaler2D, MinMaxSeqBySeqScaler3D, MaxSeqBySeqScaler3D


class FeatureSetServiceTestCase(TestCase):

    def test_create_feature_set(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'do_fit_test' : False,
            'feature_set_type': 'X'
        }
        feature_set_service = FeatureSetService()
        feature_set = feature_set_service.create_feature_set(feature_set_config)
        self.assertEqual(type(feature_set.scaler), type(MeanVarianceScaler3D()))
        self.assertEqual(feature_set.feature_list, ['feature1', 'feature2'])

        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'TimeSeriesScalerMeanVariance'
            },
            'feature_list': ['feature1', 'feature2'],
            'feature_set_type': 'X'
        }
        self.assertRaises(ValueError, feature_set_service.create_feature_set, feature_set_config)

    def test_create_feature_set_with_y(self):
        feature_set_config = {
            'scaler_config':{
                'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
            },
            'feature_list': ['feature1', 'feature2'],
            'secondary_feature_list': ['feature3', 'feature4'],
            'feature_set_type': 'X',
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

class MinMaxScaler3DTestCase(TestCase):
    def test_fit(self):
        scaler = MinMaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        self.assertEqual(scaler.min_.shape, (10, 2))
        self.assertEqual(scaler.max_.shape, (10, 2))

    def test_transform(self):
        scaler = MinMaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)
        # Check that min of X_scaled along time_steps axis is 0
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=1), np.zeros((10, 2)))
        # Check that max of X_scaled along time_steps axis is 1
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=1), np.ones((10, 2)))

    def test_inverse_transform(self):
        scaler = MinMaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, X.shape)
        np.testing.assert_almost_equal(X, X_inverse)

    def test_serialize(self):
        scaler = MinMaxSeqBySeqScaler3D()
        config = scaler.serialize()
        self.assertEqual(config, {'min': [], 'max': []})

        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()
        self.assertEqual(len(config['min']), 10)
        self.assertEqual(len(config['max']), 10)

        try:
            json.dumps(config)
        except Exception as e:
            self.fail("json.dump raised an exception unexpectedly! " + str(e))

    def test_deserialize(self):
        scaler = MinMaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()

        scaler2 = MinMaxSeqBySeqScaler3D()
        scaler2.deserialize(config)
        np.testing.assert_almost_equal(scaler.min_, scaler2.min_)
        np.testing.assert_almost_equal(scaler.max_, scaler2.max_)

        X_scaled = scaler2.transform(X)
        X_inverse = scaler2.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inverse)


class MaxScaler3DTestCase(TestCase):
    def test_fit(self):
        scaler = MaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        self.assertEqual(scaler.max_.shape, (10,))

    def test_transform(self):
        scaler = MaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)
        # Check that max of X_scaled is 1 for each element
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=(1,2)), np.ones(10))

    def test_inverse_transform(self):
        scaler = MaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, X.shape)
        np.testing.assert_almost_equal(X, X_inverse)

    def test_serialize(self):
        scaler = MaxSeqBySeqScaler3D()
        config = scaler.serialize()
        self.assertEqual(config, {'max': []})

        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()
        self.assertEqual(len(config['max']), 10)

        try:
            json.dumps(config)
        except Exception as e:
            self.fail("json.dumps raised an exception unexpectedly! " + str(e))

    def test_deserialize(self):
        scaler = MaxSeqBySeqScaler3D()
        X = np.random.rand(10, 5, 2)
        scaler.fit(X)
        config = scaler.serialize()

        scaler2 = MaxSeqBySeqScaler3D()
        scaler2.deserialize(config)
        np.testing.assert_almost_equal(scaler.max_, scaler2.max_)

        X_scaled = scaler2.transform(X)
        X_inverse = scaler2.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_percent_difference(self):
        scaler = MaxSeqBySeqScaler3D()
        X = np.array([[[50, 100], [25, 50]]])  # Shape (1, 2, 2)
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # Compute percent differences before scaling between time steps for each feature
        pd_before_feature0 = (X[0, 0, 0] - X[0, 1, 0]) / X[0, 0, 0]  # (50-25)/50 = 0.5
        pd_before_feature1 = (X[0, 0, 1] - X[0, 1, 1]) / X[0, 0, 1]  # (100-50)/100 = 0.5

        # Compute percent differences after scaling between time steps for each feature
        Xs = X_scaled[0]
        pd_after_feature0 = (Xs[0, 0] - Xs[1, 0]) / Xs[0, 0]
        pd_after_feature1 = (Xs[0, 1] - Xs[1, 1]) / Xs[0, 1]

        # Check that percent differences between time steps are preserved
        self.assertAlmostEqual(pd_before_feature0, pd_after_feature0)
        self.assertAlmostEqual(pd_before_feature1, pd_after_feature1)

        # Compute percent differences before scaling between features at each time step
        # Time step 0
        pd_before_t0 = (X[0, 0, 1] - X[0, 0, 0]) / X[0, 0, 1]  # (100 - 50)/100 = 0.5
        # Time step 1
        pd_before_t1 = (X[0, 1, 1] - X[0, 1, 0]) / X[0, 1, 1]  # (50 - 25)/50 = 0.5

        # Compute percent differences after scaling between features at each time step
        # Time step 0
        pd_after_t0 = (Xs[0, 1] - Xs[0, 0]) / Xs[0, 1]
        # Time step 1
        pd_after_t1 = (Xs[1, 1] - Xs[1, 0]) / Xs[1, 1]

        # Check that percent differences between features are preserved
        self.assertAlmostEqual(pd_before_t0, pd_after_t0)
        self.assertAlmostEqual(pd_before_t1, pd_after_t1)

class MinMaxSeqBySeqScaler2DTestCase(TestCase):
    def test_fit_2d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5)
        scaler.fit(X)
        self.assertEqual(scaler.min_.shape, (10,))
        self.assertEqual(scaler.max_.shape, (10,))

    def test_transform_2d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5)
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)
        # Check that min of X_scaled along columns axis is 0
        np.testing.assert_array_almost_equal(np.min(X_scaled, axis=1), np.zeros(10))
        # Check that max of X_scaled along columns axis is 1
        np.testing.assert_array_almost_equal(np.max(X_scaled, axis=1), np.ones(10))
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_fit_3d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5, 3)
        scaler.fit(X)
        self.assertEqual(scaler.min_.shape, (10,))
        self.assertEqual(scaler.max_.shape, (10,))

    def test_transform_3d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5, 3)
        X_scaled = scaler.fit_transform(X)
        self.assertEqual(X_scaled.shape, X.shape)
        # Reshape to 2D for checking min and max per sample
        X_scaled_reshaped = X_scaled.reshape(X_scaled.shape[0], -1)
        # Check that min of X_scaled along columns axis is 0
        np.testing.assert_array_almost_equal(np.min(X_scaled_reshaped, axis=1), np.zeros(10))
        # Check that max of X_scaled along columns axis is 1
        np.testing.assert_array_almost_equal(np.max(X_scaled_reshaped, axis=1), np.ones(10))
        # Test inverse transform
        X_inverse = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_inverse_transform_2d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5)
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, X.shape)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_inverse_transform_3d(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5, 3)
        X_scaled = scaler.fit_transform(X)
        X_inverse = scaler.inverse_transform(X_scaled)
        self.assertEqual(X_inverse.shape, X.shape)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_serialize(self):
        scaler = MinMaxSeqBySeqScaler2D()
        config = scaler.serialize()
        self.assertEqual(config, {'min': [], 'max': []})

        X = np.random.rand(10, 5)
        scaler.fit(X)
        config = scaler.serialize()
        self.assertEqual(len(config['min']), 10)
        self.assertEqual(len(config['max']), 10)

        try:
            json.dumps(config)
        except Exception as e:
            self.fail("json.dumps raised an exception unexpectedly! " + str(e))

    def test_deserialize(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5)
        scaler.fit(X)
        config = scaler.serialize()

        scaler2 = MinMaxSeqBySeqScaler2D()
        scaler2.deserialize(config)
        np.testing.assert_array_almost_equal(scaler.min_, scaler2.min_)
        np.testing.assert_array_almost_equal(scaler.max_, scaler2.max_)

        X_scaled = scaler2.transform(X)
        X_inverse = scaler2.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inverse)

    def test_transform_raises_with_unfitted_scaler(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10, 5)
        with self.assertRaises(RuntimeError):
            scaler.transform(X)

    def test_inverse_transform_raises_with_unfitted_scaler(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X_scaled = np.random.rand(10, 5)
        with self.assertRaises(RuntimeError):
            scaler.inverse_transform(X_scaled)

    def test_invalid_input_dimension(self):
        scaler = MinMaxSeqBySeqScaler2D()
        X = np.random.rand(10)
        with self.assertRaises(ValueError):
            scaler.fit(X)

        X = np.random.rand(10, 5, 3, 2)
        with self.assertRaises(ValueError):
            scaler.fit(X)