from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from django.test import TestCase

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity

from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService
from sequenceset_manager.strategy.SequenceSetStrategy import PopulateDataBundleStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy.BaseStrategy import AssignAttributesStrategy
# class CreateDataBundleStrategyTestCase(TestCase):
#     def setUp(self):
#         # Create test dataframes
#         self.df1 = pd.DataFrame({'open': [1, 2, 3], 'high': [1.1, 2.2, 3.3], 'close+1': [2, 3, 4]})
#         self.df2 = pd.DataFrame({'open': [4, 5, 6], 'high': [4.4, 5.5, 6.6], 'close+1': [5, 6, 7]})

#         # Create SequenceSet models
#         self.sequence_set_model_1 = SequenceSet.objects.create(
#             dataset_type="stock",
#             sequence_length=3,
#             start_timestamp="2022-01-01",
#             end_timestamp="2022-01-03",
#             feature_dict={"open": 0, "high": 1, "close+1": 2},
#             metadata={"ticker": "AAPL"}
#         )

#         self.sequence_set_model_2 = SequenceSet.objects.create(
#             dataset_type="stock",
#             sequence_length=3,
#             start_timestamp="2022-02-01",
#             end_timestamp="2022-02-03",
#             feature_dict={"open": 0, "high": 1, "close+1": 2},
#             metadata={"ticker": "GOOGL"}
#         )

#         # Assign sequences to the SequenceSets
#         sequences_1 = SequenceSetService.create_sequence_objects(self.sequence_set_model_1, self.df1)
#         sequences_2 = SequenceSetService.create_sequence_objects(self.sequence_set_model_2, self.df2)

#         # Convert SequenceSets to entities
#         self.sequence_set_entity_1 = SequenceSetEntity.from_db(self.sequence_set_model_1)
#         self.sequence_set_entity_2 = SequenceSetEntity.from_db(self.sequence_set_model_2)

#         self.sequence_set_entity_1.sequences = sequences_1
#         self.sequence_set_entity_2.sequences = sequences_2

#         # Initialize StrategyExecutor
#         self.strategy_executor = StrategyExecutor()

#         # Initialize CreateDataBundleStrategy
#         self.strategy_request = StrategyRequestEntity()
#         self.strategy = CreateDataBundleStrategy(self.strategy_executor, self.strategy_request)

#         # Combine SequenceSetEntities into a list
#         self.sequence_sets = [self.sequence_set_entity_1, self.sequence_set_entity_2]

#     def test_apply_creates_and_assigns_data_bundle(self):
#         # Execute the strategy
#         self.strategy.apply(self.sequence_sets)

#         for sequence_set in self.sequence_sets:
#             # Verify that a DataBundleEntity was created and assigned
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             self.assertIsNotNone(data_bundle)
#             self.assertIsInstance(data_bundle, DataBundleEntity)

#     def test_create_bundle_creates_data_bundle_entity(self):
#         # Test the `create_bundle` method
#         data_bundle = self.strategy.create_bundle(self.sequence_set_entity_1)

#         # Verify that a DataBundleEntity is created
#         self.assertIsNotNone(data_bundle)
#         self.assertIsInstance(data_bundle, DataBundleEntity)

#     def test_verify_executable_raises_not_implemented(self):
#         # Test that `verify_executable` raises NotImplementedError
#         with self.assertRaises(NotImplementedError):
#             self.strategy.verify_executable(self.sequence_set_entity_1, self.strategy_request)




class PopulateDataBundleStrategyTestCase(TestCase):
    def setUp(self):
        # Create test dataframes
        self.df1 = pd.DataFrame({'open': [1, 2, 3, 4, 5], 'high': [2, 1.5, 2.5, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        self.df2 = pd.DataFrame({'high': [2, 1.5, 2.5, 4, 5],'open': [1, 2, 3, 4, 5], 'close+1': [2, 3, 4, 5, 6]})

        # Create SequenceSet models
        self.sequence_set_model_1 = SequenceSet.objects.create(
            dataset_type='stock',
            sequence_length=2,
            start_timestamp='2022-01-01',
            end_timestamp='2022-01-04',
            feature_dict={'open': 0, 'high': 1, 'close+1': 2},
            metadata={'ticker': 'AAPL'}
        )

        self.sequence_set_model_2 = SequenceSet.objects.create(
            dataset_type='stock',
            sequence_length=2,
            start_timestamp='2022-02-01',
            end_timestamp='2022-02-04',
            feature_dict={'open': 0, 'high': 1, 'close+1': 2},
            metadata={'ticker': 'GOOGL'}
        )

        # Assign sequences to the SequenceSets
        sequences_1 = SequenceSetService.create_sequence_objects(self.sequence_set_model_1, self.df1)
        sequences_1 = sorted(sequences_1, key=lambda x: x.start_timestamp)
        sequences_2 = SequenceSetService.create_sequence_objects(self.sequence_set_model_2, self.df2)
        sequences_2 = sorted(sequences_2, key=lambda x: x.start_timestamp)

        # Convert SequenceSets to entities
        self.sequence_set_entity_1 = SequenceSetEntity.from_db(self.sequence_set_model_1)
        self.sequence_set_entity_2 = SequenceSetEntity.from_db(self.sequence_set_model_2)

        self.sequence_set_entity_1.set_attribute('sequences', sequences_1)
        self.sequence_set_entity_2.set_attribute('sequences', sequences_2)

        X_features = ['open', 'high']
        y_features = ['close+1']

        self.sequence_set_entity_1.X_features = X_features
        self.sequence_set_entity_1.y_features = y_features
        self.sequence_set_entity_2.X_features = X_features
        self.sequence_set_entity_2.y_features = y_features

        # Initialize StrategyExecutor and register nested strategy
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor.register_strategy(
            AssignAttributesStrategy.__name__, AssignAttributesStrategy
        )

        # Initialize PopulateDataBundleStrategy
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.param_config = {}
        self.strategy = PopulateDataBundleStrategy(self.strategy_executor, self.strategy_request)

        self.sequence_set_entity_1.add_child(DataBundleEntity())
        self.sequence_set_entity_1.set_attribute('X_features', X_features)
        self.sequence_set_entity_1.set_attribute('y_features', y_features)
        self.sequence_set_entity_2.add_child(DataBundleEntity())
        self.sequence_set_entity_2.set_attribute('X_features', X_features)
        self.sequence_set_entity_2.set_attribute('y_features', y_features)

        self.sequence_set_entity_1.set_attribute('seq_end_dates', [seq.end_timestamp for seq in sequences_1])
        self.sequence_set_entity_2.set_attribute('seq_end_dates', [seq.end_timestamp for seq in sequences_2])

        # Combine SequenceSetEntities into a list
        self.sequence_sets = [self.sequence_set_entity_1, self.sequence_set_entity_2]

    def test_apply_populates_existing_data_bundles(self):
        # Execute the strategy
        self.strategy.apply(self.sequence_sets)

        for sequence_set in self.sequence_sets:
            # Verify that the data_bundle is populated
            data_bundle = sequence_set.get_children_by_type(EntityEnum.DATA_BUNDLE)
            self.assertEqual(len(data_bundle), 1)
            data_bundle = data_bundle[0]

            # Verify the nested strategy was executed
            self.assertIn("X", data_bundle.get_attributes())
            self.assertIn("y", data_bundle.get_attributes())
            self.assertIn("row_ids", data_bundle.get_attributes())
            self.assertIn("X_feature_dict", data_bundle.get_attributes())
            self.assertIn("y_feature_dict", data_bundle.get_attributes())

    def test_create_feature_dict(self):
        # Test feature dictionary creation
        X_features = ['open', 'high']
        y_features = ['close+1']

        feature_dict = self.strategy.create_feature_dict(X_features, y_features)
        expected_dict = {'open': 0, 'high': 1, 'close+1': 2}

        self.assertEqual(feature_dict, expected_dict)

    def test_create_3d_array_seq(self):
        # Test 3D array creation
        X_features = ['open', 'high']
        y_features = ['close+1']
        feature_dict = self.strategy.create_feature_dict(X_features, y_features)

        X, y, row_ids = self.strategy.create_3d_array_seq(
            self.sequence_set_entity_1, X_features, y_features, feature_dict
        )

        expected_X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        expected_y = np.array([[[3]], [[4]], [[5]], [[6]]])
        expected_row_ids = [seq.id for seq in self.sequence_set_entity_1.get_attribute('sequences')]

        np.testing.assert_almost_equal(X, expected_X)
        np.testing.assert_almost_equal(y, expected_y)
        self.assertEqual(row_ids, expected_row_ids)

    def test_verify_executable_raises_with_no_databundle(self):
        self.sequence_set_entity_2.children = []
        with self.assertRaises(ValueError):
            self.strategy.verify_executable(self.sequence_sets, self.strategy_request)

# class SplitAllBundlesDataStrategyTestCase(TestCase):
#     def setUp(self):
#         # Mock strategy executor
#         self.strategy_executor = StrategyExecutor()
#         self.strategy_executor.register_strategy(SplitBundleDateStrategy.__name__, SplitBundleDateStrategy)

#         # Create sequence set entities with sequences having end_timestamps
#         self.sequence_sets = []
#         self.split_date = pd.Timestamp('2020-01-03')

#         for i in range(2):  # Create 2 sequence sets for testing
#             sequence_set = SequenceSetEntity()
#             sequence_set.sequences = [
#                 self.create_mock_sequence(pd.Timestamp('2020-01-01')),
#                 self.create_mock_sequence(pd.Timestamp('2020-01-02')),
#                 self.create_mock_sequence(pd.Timestamp('2020-01-03')),
#                 self.create_mock_sequence(pd.Timestamp('2020-01-04'))
#             ]
#             data_bundle = DataBundleEntity()
#             data_bundle.set_dataset({
#                 'X': np.random.rand(4, 2, 2),  # Example dataset
#                 'y': np.random.rand(4, 1, 1),
#                 'row_ids': [1, 2, 3, 4]
#             })
#             sequence_set.set_entity_map({EntityEnum.DATA_BUNDLE.value: data_bundle})
#             self.sequence_sets.append(sequence_set)

#         # Create strategy request
#         self.strategy_request = StrategyRequestEntity()
#         self.strategy_request.param_config = {
#             'split_date': self.split_date
#         }

#         # Initialize SplitAllBundlesDataStrategy
#         self.strategy = SplitAllBundlesDataStrategy(self.strategy_executor, self.strategy_request)

#     def create_mock_sequence(self, end_timestamp):
#         """
#         Creates a mock sequence with an end_timestamp.
#         """
#         class MockSequence:
#             def __init__(self, end_timestamp):
#                 self.end_timestamp = end_timestamp

#         return MockSequence(end_timestamp)

#     def test_apply_splits_all_bundles(self):
#         # Apply the strategy
#         self.strategy.apply(self.sequence_sets)

#         for sequence_set in self.sequence_sets:
#             # Verify that DataBundleEntity exists in the SequenceSetEntity
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             self.assertIsNotNone(data_bundle)

#             # Verify that the split keys are present in the dataset
#             dataset = data_bundle.dataset
#             self.assertIn('X_train', dataset)
#             self.assertIn('X_test', dataset)
#             self.assertIn('y_train', dataset)
#             self.assertIn('y_test', dataset)

#     def test_verify_executable(self):
#         # Test valid configuration
#         try:
#             self.strategy.verify_executable(self.sequence_sets, self.strategy_request)
#         except Exception as e:
#             self.fail(f"verify_executable raised an unexpected exception: {e}")

#         # Test missing split_date
#         invalid_request = StrategyRequestEntity()
#         invalid_request.param_config = {}
#         with self.assertRaises(ValueError) as context:
#             self.strategy.verify_executable(self.sequence_sets, invalid_request)
#         self.assertEqual(str(context.exception), "Missing split_date in config")

#         # Test missing DataBundleEntity in a SequenceSetEntity
#         self.sequence_sets[0].entity_map = {}
#         with self.assertRaises(ValueError) as context:
#             self.strategy.verify_executable(self.sequence_sets, self.strategy_request)
#         self.assertEqual(str(context.exception), "DataBundle not found in SequenceSetEntity")

#     def test_create_strategy_request(self):
#         # Test strategy request creation
#         sequence_set = self.sequence_sets[0]
#         dates = [sequence.end_timestamp for sequence in sequence_set.sequences]
#         strategy_request = self.strategy.create_strategy_request(dates, self.split_date)

#         # Verify strategy request attributes
#         self.assertEqual(strategy_request.strategy_name, SplitBundleDateStrategy.__name__)
#         self.assertIsNone(strategy_request.strategy_path)
#         self.assertIn('split_date', strategy_request.param_config)
#         self.assertIn('date_list', strategy_request.param_config)
#         self.assertEqual(strategy_request.param_config['split_date'], self.split_date)
#         self.assertEqual(strategy_request.param_config['date_list'], dates)


# class ScaleSeqSetsByFeaturesStrategyTestCase(TestCase):
#     def setUp(self):
#         # Initialize StrategyExecutor and StrategyRequestEntity
#         self.strategy_executor = StrategyExecutor()
#         self.strategy_executor.register_strategy(CreateFeatureSetsStrategy.__name__, CreateFeatureSetsStrategy)
#         self.strategy_executor.register_strategy(ScaleByFeatureSetsStrategy.__name__, ScaleByFeatureSetsStrategy)
#         self.strategy_request = StrategyRequestEntity()

#         # Define feature set configs for the test
#         self.feature_set_configs = [
#             {
#                 'scaler_config': {'scaler_name': 'MEAN_VARIANCE_SCALER_3D'},
#                 'feature_list': ['open', 'high'],
#                 'feature_set_type': 'X',
#                 'do_fit_test': False
#             },
#             {
#                 'scaler_config': {'scaler_name': 'TIME_STEP_SCALER_3D'},
#                 'feature_list': ['close+1'],
#                 'feature_set_type': 'y',
#                 'do_fit_test': False
#             }
#         ]

#         # Mock sequence sets with data bundles and feature sets
#         self.sequence_sets = []
#         for _ in range(2):  # Create two SequenceSetEntities
#             sequence_set = SequenceSetEntity()

#             # Mock DataBundleEntity and dataset
#             data_bundle = DataBundleEntity()
#             data_bundle.dataset = {
#                 'X_train': np.random.rand(10, 5, 3),  # Mock training data
#                 'X_test': np.random.rand(5, 5, 3),   # Mock test data
#                 'y_train': np.random.rand(10, 5, 1), # Mock training targets
#                 'y_test': np.random.rand(5, 5, 1),  # Mock test targets
#                 'X_feature_dict': {'open': 0, 'high': 1, 'low': 2},
#                 'y_feature_dict': {'close+1': 0}
#             }

#             # Add the DataBundleEntity to the SequenceSetEntity
#             sequence_set.set_entity_map({EntityEnum.DATA_BUNDLE.value: data_bundle})

#             # Create and apply feature sets to the data bundle using CreateFeatureSetsStrategy
#             create_feature_set_request = StrategyRequestEntity()
#             create_feature_set_request.param_config = {'feature_set_configs': self.feature_set_configs}
#             create_feature_set_strategy = CreateFeatureSetsStrategy(self.strategy_executor, create_feature_set_request)
#             create_feature_set_strategy.apply(data_bundle)

#             self.sequence_sets.append(sequence_set)

#         # Initialize the ScaleSeqSetsByFeaturesStrategy
#         self.strategy = ScaleSeqSetsByFeaturesStrategy(self.strategy_executor, self.strategy_request)

#     def test_apply(self):
#         # Apply the ScaleSeqSetsByFeaturesStrategy
#         self.strategy.apply(self.sequence_sets)

#         # Verify that scaled datasets exist in each DataBundleEntity
#         for sequence_set in self.sequence_sets:
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             dataset = data_bundle.dataset

#             # Verify that scaled data is present
#             self.assertIn('X_train_scaled', dataset)
#             self.assertIn('X_test_scaled', dataset)
#             self.assertIn('y_train_scaled', dataset)
#             self.assertIn('y_test_scaled', dataset)

#             # Verify shapes match original data
#             self.assertEqual(dataset['X_train_scaled'].shape, dataset['X_train'].shape)
#             self.assertEqual(dataset['X_test_scaled'].shape, dataset['X_test'].shape)
#             self.assertEqual(dataset['y_train_scaled'].shape, dataset['y_train'].shape)
#             self.assertEqual(dataset['y_test_scaled'].shape, dataset['y_test'].shape)

#             # Verify that scaling occurred
#             np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, dataset['X_train'], dataset['X_train_scaled'])
#             np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, dataset['X_test'], dataset['X_test_scaled'])
#             np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, dataset['y_train'], dataset['y_train_scaled'])
#             np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, dataset['y_test'], dataset['y_test_scaled'])

#     def test_verify_executable(self):
#         # Ensure no exception is raised when data bundles are present
#         try:
#             self.strategy.verify_executable(self.sequence_sets, self.strategy_request)
#         except Exception as e:
#             self.fail(f"verify_executable raised an unexpected exception: {e}")

#         # Test behavior when a sequence set lacks a data bundle
#         sequence_set_without_bundle = SequenceSetEntity()
#         self.sequence_sets.append(sequence_set_without_bundle)

#         with self.assertRaises(ValueError) as context:
#             self.strategy.verify_executable(self.sequence_sets, self.strategy_request)
#         self.assertIn("DataBundle not found", str(context.exception))



# class CombineSeqBundlesStrategyTestCase(TestCase):
#     def setUp(self):
#         # Mock StrategyExecutor
#         self.strategy_executor = MagicMock()

#         # Mock StrategyRequestEntity
#         self.strategy_request = StrategyRequestEntity()

#         # Create sample sequence sets with data bundles
#         self.sequence_sets = []

#         for i in range(3):  # Three sequence sets
#             sequence_set = SequenceSetEntity()
#             data_bundle = DataBundleEntity()

#             # Populate dataset for the data bundle
#             dataset = {
#                 'X_train': np.random.rand(5, 10, 2) + i,  # Differentiating by i
#                 'y_train': np.random.rand(5, 1) + i,
#                 'X_test': np.random.rand(2, 10, 2) + i,
#                 'y_test': np.random.rand(2, 1) + i,
#                 'X_feature_dict': {'feature1': 0, 'feature2': 1},
#                 'y_feature_dict': {'output': 0}
#             }
#             data_bundle.set_dataset(dataset)
#             data_bundle.set_entity_map({EntityEnum.FEATURE_SETS.value: []})
#             sequence_set.set_entity_map({EntityEnum.DATA_BUNDLE.value: data_bundle})

#             self.sequence_sets.append(sequence_set)

#         # Mock the CombineDataBundlesStrategy execution to return a combined data bundle
#         def mock_execute(data_bundles, strategy_request):
#             combined_dataset = {}
#             keys = data_bundles[0].dataset.keys()

#             for key in keys:
#                 if key in ['X_feature_dict', 'y_feature_dict']:
#                     combined_dataset[key] = data_bundles[0].dataset[key]
#                 else:
#                     combined_dataset[key] = np.concatenate(
#                         [bundle.dataset[key] for bundle in data_bundles], axis=0
#                     )

#             combined_bundle = DataBundleEntity()
#             combined_bundle.set_dataset(combined_dataset)
#             combined_bundle.set_entity_map({EntityEnum.FEATURE_SETS.value: data_bundles[0].get_entity(EntityEnum.FEATURE_SETS.value)})

#             strategy_request.ret_val['data_bundle'] = combined_bundle
#             return strategy_request

#         self.strategy_executor.execute = MagicMock(side_effect=mock_execute)

#         # Instantiate CombineSeqBundlesStrategy
#         self.strategy = CombineSeqBundlesStrategy(self.strategy_executor, self.strategy_request)

#     def test_apply_combines_data_bundles(self):
#         # Apply the strategy
#         self.strategy.apply(self.sequence_sets)

#         # Retrieve the combined data bundle from ret_val
#         combined_bundle = self.strategy_request.ret_val['data_bundle']

#         # Assert combined data bundle is not None
#         self.assertIsNotNone(combined_bundle)

#         # Assert dataset keys match original datasets
#         original_keys = self.sequence_sets[0].get_entity(EntityEnum.DATA_BUNDLE.value).dataset.keys()
#         self.assertSetEqual(set(combined_bundle.dataset.keys()), set(original_keys))

#         # Assert combined shapes match concatenated data
#         expected_X_train_shape = sum(bundle.dataset['X_train'].shape[0] for bundle in
#             [seq_set.get_entity(EntityEnum.DATA_BUNDLE.value) for seq_set in self.sequence_sets])
#         self.assertEqual(combined_bundle.dataset['X_train'].shape[0], expected_X_train_shape)

#         expected_y_train_shape = sum(bundle.dataset['y_train'].shape[0] for bundle in
#             [seq_set.get_entity(EntityEnum.DATA_BUNDLE.value) for seq_set in self.sequence_sets])
#         self.assertEqual(combined_bundle.dataset['y_train'].shape[0], expected_y_train_shape)

#         # Verify other keys remain consistent
#         self.assertEqual(combined_bundle.dataset['X_feature_dict'],
#             self.sequence_sets[0].get_entity(EntityEnum.DATA_BUNDLE.value).dataset['X_feature_dict'])
#         self.assertEqual(combined_bundle.dataset['y_feature_dict'],
#             self.sequence_sets[0].get_entity(EntityEnum.DATA_BUNDLE.value).dataset['y_feature_dict'])
