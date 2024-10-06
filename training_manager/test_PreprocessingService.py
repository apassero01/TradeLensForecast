from copy import deepcopy

import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService
from training_manager.PreprocessingService import PreprocessingService
from training_manager.models import FeatureSet
from django.test import TestCase


class PreprocessingServiceTestCase(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'open': [1, 2, 3, 4, 5], 'high' : [2,1.5,2.5,4,5], 'close+1': [2, 3, 4, 5, 6]})
        self.sequence_set = SequenceSet.objects.create(dataset_type='stock', sequence_length=2, start_timestamp='2020-01-01', end_timestamp='2020-01-05', feature_dict={'open': 0, 'high': 1, 'close+1': 2}, metadata={'ticker': 'AAPL'})
        self.stock_sequences = SequenceSetService.create_sequence_objects(self.sequence_set, self.df)
        self.stock_sequences = sorted(self.stock_sequences, key=lambda x: x.start_timestamp)

    def test_create_3d_array(self):
        X_features = ['open', 'high']
        y_features = ['close+1']
        feature_dict = self.sequence_set.feature_dict
        service = PreprocessingService()
        X, y, sequence_ids = service.create_3d_array(self.stock_sequences, X_features, y_features, feature_dict)
        expected_X = np.array([[[1, 2], [2, 1.5]], [[2, 1.5], [3, 2.5]], [[3, 2.5], [4, 4]], [[4, 4], [5, 5]]])
        expected_y = np.array([[[3]], [[4]], [[5]], [[6]]])
        np.testing.assert_almost_equal(X, expected_X)
        np.testing.assert_almost_equal(y, expected_y)

    def test_scale(self):
        arr1 = np.random.rand(10, 5, 2)
        arr2 = np.random.rand(15, 5, 2)

        scaler = TimeSeriesScalerMeanVariance()

        service = PreprocessingService()

        arr1_scaled = service.scale(deepcopy(scaler), arr1)

        np.testing.assert_almost_equal(arr1_scaled, scaler.fit_transform(arr1))

        arr1_scaled, arr2_scaled = service.scale(deepcopy(scaler), arr1, arr2)
        np.testing.assert_almost_equal(arr1_scaled, scaler.fit_transform(arr1))
        np.testing.assert_almost_equal(arr2_scaled, scaler.transform(arr2))

    def test_scale_by_features(self):
        arr1 = np.random.rand(10, 5, 2)
        arr2 = np.random.rand(15, 5, 2)

        scaler1 = TimeSeriesScalerMeanVariance()
        scaler2 = TimeSeriesScalerMinMax()
        feature_sets = []
        feature_sets.append(FeatureSet(scaler_config={}, feature_list=['open']))
        feature_sets[0].scaler = deepcopy(scaler1)
        feature_sets.append(FeatureSet(scaler_config={}, feature_list=['high']))
        feature_sets[1].scaler = deepcopy(scaler2)

        feature_dict = {'open': 0, 'high': 1}

        service = PreprocessingService()
        arr1_scaled = service.scale_by_features(feature_sets, feature_dict, arr1)

        np.testing.assert_almost_equal(arr1_scaled[:,:,0:1], scaler1.fit_transform(arr1[:,:,0]))
        np.testing.assert_almost_equal(arr1_scaled[:,:,1:], scaler2.fit_transform(arr1[:,:,1]))

        arr1_scaled, arr2_scaled = service.scale_by_features(feature_sets, feature_dict, arr1, arr2)
        np.testing.assert_almost_equal(arr1_scaled[:,:,0:1], scaler1.fit_transform(arr1[:,:,0]))
        np.testing.assert_almost_equal(arr1_scaled[:,:,1:], scaler2.fit_transform(arr1[:,:,1]))
        np.testing.assert_almost_equal(arr2_scaled[:,:,0:1], scaler1.transform(arr2[:,:,0]))
        np.testing.assert_almost_equal(arr2_scaled[:,:,1:], scaler2.transform(arr2[:,:,1]))


    def test_train_test_split(self):
        X = np.random.rand(5, 5, 2)
        y = np.random.rand(5, 1, 1)
        sequence_ids = [1, 2, 3, 4, 5]
        dates = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        split_date = '2020-01-04'

        service = PreprocessingService()
        X_train, X_test, y_train, y_test, train_seq_ids, test_seq_ids = service.train_test_split(X, y, dates, split_date, sequence_ids)

        self.assertEqual(X_train.shape[0], 3)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(y_train.shape[0], 3)
        self.assertEqual(y_test.shape[0], 2)
        self.assertEqual(dates.index(split_date), 3)
        self.assertEqual(train_seq_ids, sequence_ids[:3])
        self.assertEqual(test_seq_ids, sequence_ids[3:])


        dates = ['2020-01-01', '2020-01-02', '2020-01-05', '2020-01-06', '2020-01-07']
        split_date = '2020-01-04'

        X_train, X_test, y_train, y_test, train_seq_ids, test_seq_ids = service.train_test_split(X, y, dates, split_date, sequence_ids)

        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_test.shape[0], 3)
        self.assertEqual(y_train.shape[0], 2)
        self.assertEqual(y_test.shape[0], 3)
        self.assertEqual(train_seq_ids, sequence_ids[:2])
        self.assertEqual(test_seq_ids, sequence_ids[2:])

    def test_combine_seq_sets(self):
        X1 = np.random.rand(5, 5, 2)
        y1 = np.random.rand(5, 1, 1)
        X2 = np.random.rand(5, 5, 2)
        y2 = np.random.rand(5, 1, 1)
        sequence_ids1 = np.array([1, 2, 3, 4, 5])
        sequence_ids2 = np.array([6, 7, 8, 9, 10])

        service = PreprocessingService()

        X1Train, X1Test, y1Train, y1Test, train_seq_ids1, test_seq_ids1 = service.train_test_split(X1, y1, ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'], '2020-01-04', sequence_ids1)
        X2Train, X2Test, y2Train, y2Test, train_seq_ids2, test_seq_ids2 = service.train_test_split(X2, y2, ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'], '2020-01-04', sequence_ids2)

        seq_set1 = deepcopy(self.sequence_set)
        seq_set2 = deepcopy(self.sequence_set)

        seq_set1.X_train = X1Train
        seq_set1.y_train = y1Train
        seq_set1.X_test = X1Test
        seq_set1.y_test = y1Test
        seq_set1.train_seq_ids = train_seq_ids1
        seq_set1.test_seq_ids = test_seq_ids1

        seq_set2.X_train = X2Train
        seq_set2.y_train = y2Train
        seq_set2.X_test = X2Test
        seq_set2.y_test = y2Test
        seq_set2.train_seq_ids = train_seq_ids2
        seq_set2.test_seq_ids = test_seq_ids2

        X_train, X_test, y_train, y_test, train_seq_ids, test_seq_ids = service.combine_seq_sets([seq_set1, seq_set2])

        self.assertEqual(X_train.shape[0], 6)
        self.assertEqual(X_test.shape[0], 4)
        self.assertEqual(y_train.shape[0], 6)
        self.assertEqual(y_test.shape[0], 4)

        self.assertEqual(len(train_seq_ids), 6)
        self.assertEqual(len(test_seq_ids), 4)
        np.testing.assert_array_equal(train_seq_ids, np.concatenate([train_seq_ids1, train_seq_ids2]))
        np.testing.assert_array_equal(test_seq_ids, np.concatenate([test_seq_ids1, test_seq_ids2]))

