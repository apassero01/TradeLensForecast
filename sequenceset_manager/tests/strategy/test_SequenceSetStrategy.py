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
from shared_utils.entities.service.EntityService import EntityService
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

        self.entity_service = EntityService()

        self.data_bundle_entity1 = DataBundleEntity()
        self.data_bundle_entity2 = DataBundleEntity()
        self.entity_service.save_entity(self.data_bundle_entity1)
        self.entity_service.save_entity(self.data_bundle_entity2)
        self.sequence_set_entity_1.add_child(self.data_bundle_entity1)
        self.sequence_set_entity_1.set_attribute('X_features', X_features)
        self.sequence_set_entity_1.set_attribute('y_features', y_features)
        self.sequence_set_entity_2.add_child(self.data_bundle_entity2)
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
            data_bundle_ids = self.entity_service.get_children_ids_by_type(sequence_set, EntityEnum.DATA_BUNDLE)
            self.assertEqual(len(data_bundle_ids), 1)
            data_bundle = self.entity_service.get_entity(data_bundle_ids[0])


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