from copy import deepcopy

import numpy as np
import pandas as pd
from django.test import TestCase

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.entities.FeatureSetEntity import FeatureSetEntity
from data_bundle_manager.strategy.DataBundleStrategy import CreateFeatureSetsStrategy, \
    SplitBundleDateStrategy, ScaleByFeatureSetsStrategy, CombineDataBundlesStrategy
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor


class CreateFeatureSetsStrategyTestCase(TestCase):
    def setUp(self):
        # Mock strategy request with configuration
        self.strategy_request = StrategyRequestEntity()
        self.entity_service = EntityService()
        self.strategy_request.param_config = {
            'feature_set_configs': [
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
                    'feature_list': ['open', 'high'],
                    'feature_set_type': 'X',
                    'do_fit_test': False,
                    'secondary_feature_list': None
                },
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
                    'feature_list': ['close+1'],
                    'feature_set_type': 'y',
                    'do_fit_test': False,
                    'secondary_feature_list': None
                }
            ]
        }

        self.strategy_executor = StrategyExecutor()
        self.create_feature_sets_strategy = CreateFeatureSetsStrategy(
            self.strategy_executor,
            self.strategy_request
        )
        self.data_bundle = DataBundleEntity()
        self.data_bundle.set_attribute("X_feature_dict", {'open': 0, 'high': 1})
        self.data_bundle.set_attribute("y_feature_dict", {'close+1': 0})

    def test_apply_creates_feature_sets(self):
        # Apply the strategy
        self.create_feature_sets_strategy.apply(self.data_bundle)

        # Get the feature sets from the data bundle
        feature_set_ids = self.entity_service.get_children_ids_by_type(self.data_bundle, EntityEnum.FEATURE_SET)
        feature_sets = [self.entity_service.get_entity(fs_id) for fs_id in feature_set_ids]

        # Verify the correct number of feature sets were created
        self.assertEqual(len(feature_sets), 2)

        # Check the first feature set
        feature_set_1 = feature_sets[0]
        self.assertIsInstance(feature_set_1, FeatureSetEntity)
        self.assertEqual(feature_set_1.feature_list, ['open', 'high'])
        self.assertEqual(feature_set_1.feature_set_type, 'X')
        self.assertEqual(feature_set_1.do_fit_test, False)

        # Check the second feature set
        feature_set_2 = feature_sets[1]
        self.assertIsInstance(feature_set_2, FeatureSetEntity)
        self.assertEqual(feature_set_2.feature_list, ['close+1'])
        self.assertEqual(feature_set_2.feature_set_type, 'y')
        self.assertEqual(feature_set_2.do_fit_test, False)

    def test_recreate_feature_sets_should_have_same_ids(self):
        self.create_feature_sets_strategy.apply(self.data_bundle)

        # Get the feature sets from the data bundle
        feature_set_ids = self.entity_service.get_children_ids_by_type(self.data_bundle, EntityEnum.FEATURE_SET)

        # Apply the strategy again
        self.create_feature_sets_strategy.apply(self.data_bundle)
        new_feature_set_ids = self.entity_service.get_children_ids_by_type(self.data_bundle, EntityEnum.FEATURE_SET)

        # Verify that the feature set IDs are the same
        self.assertListEqual(feature_set_ids, new_feature_set_ids)

    def test_verify_executable_raises_error_on_invalid_config(self):
        # Test missing `feature_set_configs`
        invalid_request = StrategyRequestEntity()
        invalid_request.param_config = {}
        with self.assertRaises(ValueError) as context:
            self.create_feature_sets_strategy.verify_executable(self.data_bundle, invalid_request)
        self.assertEqual(str(context.exception), "Missing feature_set_configs in config")

        # Test missing keys in `feature_set_configs`
        invalid_request.param_config = {
            'feature_set_configs': [
                {
                    'feature_list': ['open', 'high'],  # Missing other keys
                }
            ]
        }
        with self.assertRaises(ValueError) as context:
            self.create_feature_sets_strategy.verify_executable(self.data_bundle, invalid_request)
        self.assertTrue("Missing feature_set_type in feature_set_config" in str(context.exception))

    def test_verify_executable_passes_with_valid_config(self):
        # Test valid configuration
        try:
            self.create_feature_sets_strategy.verify_executable(self.data_bundle, self.strategy_request)
        except Exception as e:
            self.fail(f"verify_executable raised an unexpected exception: {e}")

    def test_get_request_config(self):
        # Test the static `get_request_config` method
        request_config = CreateFeatureSetsStrategy.get_request_config()

        self.assertIn('strategy_name', request_config)
        self.assertEqual(request_config['strategy_name'], CreateFeatureSetsStrategy.__name__)
        self.assertIn('param_config', request_config)
        self.assertIn('feature_set_configs', request_config['param_config'])

        feature_set_config = request_config['param_config']['feature_set_configs'][0]
        self.assertIn('scaler_config', feature_set_config)
        self.assertIn('feature_list', feature_set_config)
        self.assertIn('feature_set_type', feature_set_config)
        self.assertIn('do_fit_test', feature_set_config)


class SplitBundleDateStrategyTestCase(TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']).tolist()

        self.data_bundle = DataBundleEntity()
        X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]], [[5, 5], [6, 6]]])
        y = np.array([[[3]], [[4]], [[5]], [[6]], [[7]]])
        row_ids = [1, 2, 3, 4, 5]
        
        self.data_bundle.set_attribute('X', X)
        self.data_bundle.set_attribute('y', y)
        self.data_bundle.set_attribute('row_ids', row_ids)
        self.data_bundle.date_list = self.dates

        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.param_config = {
            'split_date': pd.Timestamp('2020-01-03'),
        }

        self.data_bundle.set_attribute('seq_end_dates', self.dates)

        self.strategy_executor = StrategyExecutor()
        self.strategy = SplitBundleDateStrategy(self.strategy_executor, self.strategy_request)

    def test_apply(self):
        # Apply the strategy
        self.strategy.apply(self.data_bundle)

        # Retrieve the split datasets from the data bundle
        X_train = self.data_bundle.get_attribute('X_train')
        X_test = self.data_bundle.get_attribute('X_test')
        y_train = self.data_bundle.get_attribute('y_train')
        y_test = self.data_bundle.get_attribute('y_test')
        train_row_ids = self.data_bundle.get_attribute('train_row_ids')
        test_row_ids = self.data_bundle.get_attribute('test_row_ids')

        # Expected results
        expected_X_train = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]]])
        expected_y_train = np.array([[[3]], [[4]], [[5]]])
        expected_X_test = np.array([[[4, 4], [5, 5]],[[5, 5], [6, 6]]])
        expected_y_test = np.array([[[6]], [[7]]])
        expected_train_row_ids = [1, 2, 3]
        expected_test_row_ids = [4,5]

        # Validate results
        np.testing.assert_almost_equal(X_train, expected_X_train)
        np.testing.assert_almost_equal(y_train, expected_y_train)
        np.testing.assert_almost_equal(X_test, expected_X_test)
        np.testing.assert_almost_equal(y_test, expected_y_test)
        self.assertEqual(train_row_ids, expected_train_row_ids)
        self.assertEqual(test_row_ids, expected_test_row_ids)

    def test_train_test_split_with_nearest_date(self):
        # Test split with a non-matching split date
        self.strategy_request.param_config['split_date'] = pd.Timestamp('2020-01-03T12:00:00')

        # Apply the strategy
        self.strategy.apply(self.data_bundle)

        # Retrieve the split datasets from the data bundle
        X_train = self.data_bundle.get_attribute('X_train')
        X_test = self.data_bundle.get_attribute('X_test')
        y_train = self.data_bundle.get_attribute('y_train')
        y_test = self.data_bundle.get_attribute('y_test')
        train_row_ids = self.data_bundle.get_attribute('train_row_ids')
        test_row_ids = self.data_bundle.get_attribute('test_row_ids')

        # Check that the split uses the closest date (2020-01-03)
        self.assertEqual(X_train.shape[0], 2)  # Two sequences in training
        self.assertEqual(X_test.shape[0], 3)   # Three sequences in testing

    def test_verify_executable_raises_error_on_invalid_config(self):
        # Test missing `split_date`
        invalid_request = StrategyRequestEntity()
        invalid_request.param_config = {
            'date_list': self.dates
        }
        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.data_bundle, invalid_request)
        self.assertEqual(str(context.exception), "Missing split_date in config")

        # Test missing `date_list`
        invalid_request.param_config = {
        }

        # Test missing `X` in dataset
        self.data_bundle._attributes = {}  # Clear attributes
        self.data_bundle.set_attribute('y', [123])
        self.data_bundle.set_attribute('row_ids', [1,2,34])
        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.data_bundle, self.strategy_request)
        self.assertEqual(str(context.exception), "Missing X in dataset")

        # Test missing `y` in dataset
        self.data_bundle._attributes = {}  # Clear attributes
        self.data_bundle.set_attribute('X', [123])
        self.data_bundle.set_attribute('row_ids', [1,2,34])
        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.data_bundle, self.strategy_request)
        self.assertEqual(str(context.exception), "Missing y in dataset")

    def test_get_request_config(self):
        request_config = SplitBundleDateStrategy.get_request_config()

        self.assertIn('strategy_name', request_config)
        self.assertEqual(request_config['strategy_name'], SplitBundleDateStrategy.__name__)
        self.assertIn('strategy_path', request_config)
        self.assertIsNone(request_config['strategy_path'])

        self.assertIn('param_config', request_config)
        param_config = request_config['param_config']
        self.assertIn('split_date', param_config)
        self.assertIsNone(param_config['split_date'])


class ScaleByFeatureSetsStrategyTestCase(TestCase):
    def setUp(self):
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor.register_strategy(CreateFeatureSetsStrategy.__name__, CreateFeatureSetsStrategy)

        self.data_bundle = DataBundleEntity()
        
        self.data_bundle.set_attribute('X_train', np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]]))
        self.data_bundle.set_attribute('X_test', np.array([[[4, 4], [5, 5]]]))
        self.data_bundle.set_attribute('y_train', np.array([[[3],[4]], [[4],[5]], [[5],[6]],[[6],[7]]]))
        self.data_bundle.set_attribute('y_test', np.array([[[6],[7]]]))
        self.data_bundle.set_attribute('row_ids', [1, 2, 3, 4, 5])
        self.data_bundle.set_attribute('X_feature_dict', {'open': 0, 'high': 1})
        self.data_bundle.set_attribute('y_feature_dict', {'close+1': 0})

        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = ScaleByFeatureSetsStrategy.__name__
        strategy_request.strategy_path = None

        self.scale_strategy = ScaleByFeatureSetsStrategy(strategy_executor=StrategyExecutor, strategy_request=strategy_request)

    def apply_create_feature_sets_strategy(self, feature_set_configs):
        # Use CreateFeatureSetsStrategy to add feature sets to the data bundle
        create_feature_sets_request = StrategyRequestEntity()
        create_feature_sets_request.strategy_name = CreateFeatureSetsStrategy.__name__
        create_feature_sets_request.param_config = {'feature_set_configs': feature_set_configs}

        self.strategy_executor.execute(self.data_bundle, create_feature_sets_request)

    def test_apply_X_feature_type(self):
        feature_set_configs = [
            {
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
                'feature_list': ['open', 'high'],
                'feature_set_type': 'X',
                'do_fit_test': False
            }
        ]
        self.apply_create_feature_sets_strategy(feature_set_configs)

        # Deep copy original data
        X_train_orig = deepcopy(self.data_bundle.get_attribute('X_train'))
        X_test_orig = deepcopy(self.data_bundle.get_attribute('X_test'))

        # Apply ScaleByFeatureSetsStrategy
        self.scale_strategy.apply(self.data_bundle)

        # Get scaled data
        X_train_scaled = self.data_bundle.get_attribute('X_train_scaled')
        X_test_scaled = self.data_bundle.get_attribute('X_test_scaled')

        # Verify the scaling
        feature_sets = self.data_bundle.get_children_by_type(EntityEnum.FEATURE_SET)
        X_feature_set = next(fs for fs in feature_sets if fs.feature_set_type == 'X')
        scaler = X_feature_set.scaler

        np.testing.assert_almost_equal(X_train_scaled, scaler.fit_transform(X_train_orig))
        np.testing.assert_almost_equal(X_test_scaled, scaler.transform(X_test_orig))

    def test_apply_Y_feature_type(self):
        feature_set_configs = [
            {
                'scaler_config': {
                    'scaler_name': 'TIME_STEP_SCALER_3D'
                },
                'feature_list': ['close+1'],
                'feature_set_type': 'y',
                'do_fit_test': False
            }
        ]
        self.apply_create_feature_sets_strategy(feature_set_configs)

        # Deep copy original data
        y_train_orig = deepcopy(self.data_bundle.get_attribute('y_train'))
        y_test_orig = deepcopy(self.data_bundle.get_attribute('y_test'))

        # Apply ScaleByFeatureSetsStrategy
        self.scale_strategy.apply(self.data_bundle)

        # Get scaled data
        y_train_scaled = self.data_bundle.get_attribute('y_train_scaled')
        y_test_scaled = self.data_bundle.get_attribute('y_test_scaled')

        # Verify the scaling
        feature_sets = self.data_bundle.get_children_by_type(EntityEnum.FEATURE_SET)
        y_feature_set = next(fs for fs in feature_sets if fs.feature_set_type == 'y')
        scaler = y_feature_set.scaler

        np.testing.assert_almost_equal(y_train_scaled, scaler.fit_transform(y_train_orig))
        np.testing.assert_almost_equal(y_test_scaled, scaler.transform(y_test_orig))

    def test_apply_multiple_X_feature_sets(self):
        feature_set_configs = [
            {
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
                'feature_list': ['open'],
                'feature_set_type': 'X',
                'do_fit_test': False
            },
            {
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
                'feature_list': ['high'],
                'feature_set_type': 'X',
                'do_fit_test': False
            }
        ]
        self.apply_create_feature_sets_strategy(feature_set_configs)

        # Deep copy original data
        X_train_orig = deepcopy(self.data_bundle.get_attribute('X_train'))
        X_test_orig = deepcopy(self.data_bundle.get_attribute('X_test'))

        # Apply ScaleByFeatureSetsStrategy
        self.scale_strategy.apply(self.data_bundle)

        # Get scaled data
        X_train_scaled = self.data_bundle.get_attribute('X_train_scaled')
        X_test_scaled = self.data_bundle.get_attribute('X_test_scaled')

        # Verify scaling for each feature set
        feature_sets = self.data_bundle.get_children_by_type(EntityEnum.FEATURE_SET)
        open_feature_set = next(fs for fs in feature_sets if fs.feature_list == ['open'])
        high_feature_set = next(fs for fs in feature_sets if fs.feature_list == ['high'])

        open_scaler = open_feature_set.scaler
        high_scaler = high_feature_set.scaler

        np.testing.assert_almost_equal(X_train_scaled[:, :, 0:1], open_scaler.fit_transform(X_train_orig[:, :, 0:1]))
        np.testing.assert_almost_equal(X_train_scaled[:, :, 1:2], high_scaler.fit_transform(X_train_orig[:, :, 1:2]))
        np.testing.assert_almost_equal(X_test_scaled[:, :, 0:1], open_scaler.transform(X_test_orig[:, :, 0:1]))
        np.testing.assert_almost_equal(X_test_scaled[:, :, 1:2], high_scaler.transform(X_test_orig[:, :, 1:2]))

    def test_applyXyFeatureType1FeatureSet(self):
        # Prepare feature set configuration
        feature_set_config = {
            'scaler_config': {
                'scaler_name': 'MIN_MAX_SEQ_BY_SEQ_2D'
            },
            'feature_list': ['open', 'high', 'close+1'],
            'secondary_feature_list': ['close+1'],
            'do_fit_test': False,
            'feature_set_type': 'Xy',
        }

        # Use the CreateFeatureSetsStrategy to set up the feature sets in the data bundle
        create_feature_sets_request = StrategyRequestEntity()
        create_feature_sets_request.strategy_name = CreateFeatureSetsStrategy.__name__
        create_feature_sets_request.param_config = {'feature_set_configs': [feature_set_config]}
        self.strategy_executor.execute(self.data_bundle, create_feature_sets_request)

        # Deep copy the original data for comparison
        X_train_orig = deepcopy(self.data_bundle.get_attribute('X_train'))
        X_test_orig = deepcopy(self.data_bundle.get_attribute('X_test'))
        y_train_orig = deepcopy(self.data_bundle.get_attribute('y_train'))
        y_test_orig = deepcopy(self.data_bundle.get_attribute('y_test'))

        self.scale_strategy.apply(self.data_bundle)

        # Retrieve scaled data
        X_train_scaled = self.data_bundle.get_attribute('X_train_scaled')
        X_test_scaled = self.data_bundle.get_attribute('X_test_scaled')
        y_train_scaled = self.data_bundle.get_attribute('y_train_scaled')
        y_test_scaled = self.data_bundle.get_attribute('y_test_scaled')

        # Retrieve the feature sets and scaler
        feature_sets = self.data_bundle.get_children_by_type(EntityEnum.FEATURE_SET)
        Xy_feature_set = next(fs for fs in feature_sets if fs.feature_set_type == 'Xy')
        scaler = Xy_feature_set.scaler

        # Verify shapes are unchanged
        self.assertEqual(X_train_scaled.shape, X_train_orig.shape)
        self.assertEqual(X_test_scaled.shape, X_test_orig.shape)
        self.assertEqual(y_train_scaled.shape, y_train_orig.shape)
        self.assertEqual(y_test_scaled.shape, y_test_orig.shape)

        # Prepare feature indices
        X_feature_dict = self.data_bundle.get_attribute('X_feature_dict')
        y_feature_dict = self.data_bundle.get_attribute('y_feature_dict')

        arr1_feature_indices = [
            X_feature_dict[feature]
            for feature in Xy_feature_set.feature_list
            if feature in X_feature_dict
        ]

        # Extract features
        arr1_features = X_test_orig[:, :, arr1_feature_indices]  # Shape: (samples, time_steps, features)
        arr1_scaled_features = X_test_scaled[:, :, arr1_feature_indices]

        arr1_scaled_features_reshaped = arr1_scaled_features.reshape(arr1_scaled_features.shape[0], -1)

        # Check inverse transformation for arr1
        arr1_inverse_flat = scaler.inverse_transform(arr1_scaled_features_reshaped)
        arr1_inverse = arr1_inverse_flat.reshape(arr1_features.shape)

        # Verify that the inverse transformed data matches the original
        np.testing.assert_array_almost_equal(arr1_features, arr1_inverse, decimal=6)


class CombineDataBundlesTestCase(TestCase):
    def setUp(self):
        self.strategy_executor = StrategyExecutor()
        self.strategy_request = StrategyRequestEntity()
        self.data_bundles = []

        for i in range(2):
            data_bundle = DataBundleEntity()
            
            data_bundle.set_attribute('X_train', np.random.rand(10, 5, 3) + i)
            data_bundle.set_attribute('X_test', np.random.rand(5, 5, 3) + i)
            data_bundle.set_attribute('y_train', np.random.rand(10, 5, 1) + i)
            data_bundle.set_attribute('y_test', np.random.rand(5, 5, 1) + i)
            data_bundle.set_attribute('X_feature_dict', {'open': 0, 'high': 1, 'low': 2})
            data_bundle.set_attribute('y_feature_dict', {'close+1': 0})
            data_bundle.set_attribute('row_ids', list(range(1, 11)))
            data_bundle.set_attribute('train_row_ids', list(range(1, 11)))
            data_bundle.set_attribute('test_row_ids', list(range(1, 6)))

            # Add a mock FeatureSetEntity
            feature_set = FeatureSetEntity()
            feature_set.feature_list = ['open', 'high', 'close+1']
            feature_set.feature_set_type = 'Xy'
            feature_set.do_fit_test = False

            data_bundle.add_child(feature_set)
            self.data_bundles.append(data_bundle)

        self.strategy_request.param_config = {
            'data_bundles': self.data_bundles
        }

        self.strategy = CombineDataBundlesStrategy(self.strategy_executor, self.strategy_request)

    def test_apply(self):
        # Apply the CombineDataBundles strategy
        combined_request = self.strategy.apply(self.data_bundles)

        # Retrieve the new combined DataBundleEntity
        combined_bundle = combined_request.ret_val[EntityEnum.DATA_BUNDLE.value]
        combined_dataset = combined_bundle.get_attributes()


        # Verify combined data shapes
        self.assertEqual(combined_dataset['X_train'].shape, (20, 5, 3))  # Combined along the first axis
        self.assertEqual(combined_dataset['X_test'].shape, (10, 5, 3))
        self.assertEqual(combined_dataset['y_train'].shape, (20, 5, 1))
        self.assertEqual(combined_dataset['y_test'].shape, (10, 5, 1))

        # Verify feature dictionaries are preserved
        self.assertEqual(combined_dataset['X_feature_dict'], self.data_bundles[0].get_attribute('X_feature_dict'))
        self.assertEqual(combined_dataset['y_feature_dict'], self.data_bundles[0].get_attribute('y_feature_dict'))

        # Verify that the FeatureSets are correctly attached
        feature_sets = combined_bundle.get_children_by_type(EntityEnum.FEATURE_SET)
        self.assertEqual(len(feature_sets), len(self.data_bundles[0].get_children_by_type(EntityEnum.FEATURE_SET)))
        np.testing.assert_almost_equal(combined_bundle.get_attributes()['row_ids'], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        np.testing.assert_almost_equal(combined_bundle.get_attributes()['train_row_ids'], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        np.testing.assert_almost_equal(combined_bundle.get_attributes()['test_row_ids'], np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))

    def test_verify_executable(self):
        # Ensure no exception is raised when data bundles are correctly configured
        try:
            self.strategy.verify_executable(self.data_bundles, self.strategy_request)
        except Exception as e:
            self.fail(f"verify_executable raised an unexpected exception: {e}")

        # Test behavior when param_config is missing 'data_bundles'

    def test_incompatible_data_bundles(self):
        # Create an incompatible DataBundleEntity with different keys
        incompatible_bundle = DataBundleEntity()
        incompatible_bundle.set_attribute('X_train_diff', np.random.rand(10, 5, 3))
        incompatible_bundle.set_attribute('y_train', np.random.rand(10, 5, 1))
        incompatible_bundle.set_attribute('X_feature_dict', {'open': 0, 'high': 1, 'low': 2})
        incompatible_bundle.set_attribute('y_feature_dict', {'close+1': 0})
        self.data_bundles.append(incompatible_bundle)

        # Verify that apply raises an exception for incompatible datasets
        with self.assertRaises(ValueError) as context:
            self.strategy.apply(self.data_bundles)
        self.assertIn("DataBundles do not have the same keys", str(context.exception))