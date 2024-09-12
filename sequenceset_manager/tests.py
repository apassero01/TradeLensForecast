from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
import json
import pandas as pd
import numpy as np
from dataset_manager.services import DatasetManagerService, DatasetTrackerService, FeatureTrackerService
from sequenceset_manager.services import SequenceSetService, StockSequenceSetService, SequenceService
import math
from sequenceset_manager.models import SequenceSet, Sequence
from datetime import datetime
import random
from copy import deepcopy

class StockSequenceSetServiceTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.ticker = 'AAPL'
        self.start_date = '2019-01-01'
        self.end_date = '2021-01-01'
        self.interval = '1d'

        response = self.client.post('/dataset_manager/create_stock_data/',
                                    data=json.dumps({
                                        'ticker': self.ticker,
                                        'start_date': self.start_date,
                                        'end_date': self.end_date,
                                        'interval': self.interval
                                    }),
                                    content_type='application/json')
        data = response.json()
        self.df = pd.DataFrame(data).T

    def test_create_sequence_objects(self):
        sequence_length = 10
        timeframe = self.interval

        sequence_set = SequenceSet.objects.create(
            dataset_type='stock',
            sequence_length=sequence_length,
            start_timestamp=datetime.strptime(self.start_date, '%Y-%m-%d'),
            end_timestamp=datetime.strptime(self.end_date, '%Y-%m-%d'),
            feature_dict=SequenceSetService.create_feature_dict(self.df),
            metadata={'dataset_type': 'stock', 'ticker': self.ticker, 'timeframe': self.interval}
        )

        sequences = SequenceSetService.create_sequence_objects(sequence_set, self.df)

        # Check the number of sequences generated
        expected_num_sequences = len(self.df) - sequence_length + 1
        self.assertEqual(len(sequences), expected_num_sequences)

        # Verify each sequence
        for seq_obj in sequences:
            # Ensure the sequence length is correct
            self.assertEqual(seq_obj.sequence_length, sequence_length)
            self.assertEqual(len(seq_obj.sequence_data), sequence_length)
            self.assertEqual(len(seq_obj.sequence_data[0]), len(self.df.columns))

            # Verify that the sequence data matches the DataFrame values
            start_index = self.df.index.get_loc(seq_obj.start_timestamp)
            end_index = self.df.index.get_loc(seq_obj.end_timestamp)

            expected_seq = self.df.iloc[start_index:end_index + 1].values.tolist()
            self.assertTrue((seq_obj.sequence_data, expected_seq))

            # Verify the column dictionary
            for feature_name, index in sequence_set.feature_dict.items():
                # Extract the slice of the 2D array corresponding to this feature
                sequence_slice = [row[index] for row in seq_obj.sequence_data]

                # Compare it to the original DataFrame column for this feature
                original_values = self.df[feature_name].iloc[start_index:end_index + 1].tolist()
                self.assertTrue(compare_lists_with_nan(sequence_slice, original_values))

        # Ensure the ticker and timeframe match
        self.assertEqual(sequence_set.metadata['ticker'], self.ticker.upper())
        self.assertEqual(sequence_set.metadata['timeframe'], timeframe)

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_create_sequence_set(self, mock_get_df_data):

        mock_get_df_data.return_value = self.df
        sequence_length = 10

        StockSequenceSetService.create_sequence_set(sequence_length = sequence_length, dataset_type = "stock", ticker = self.ticker, interval = self.interval)

        sequence_set = SequenceSet.objects.first()
        sequences = Sequence.objects.all()

        # Check the number of sequences generated
        expected_num_sequences = len(self.df) - sequence_length + 1
        self.assertEqual(len(sequences), expected_num_sequences)

        self.df.index = pd.to_datetime(self.df.index)

        # Verify each sequence
        for seq_obj in sequences:
            # Ensure the sequence length is correct
            seq_obj.sequence_data = SequenceService.transpose(seq_obj.sequence_data)
            self.assertEqual(seq_obj.sequence_length, sequence_length)
            self.assertEqual(len(seq_obj.sequence_data), sequence_length)
            self.assertEqual(len(seq_obj.sequence_data[0]), len(self.df.columns))

            # Verify that the sequence data matches the DataFrame values
            start_index = self.df.index.get_loc(seq_obj.start_timestamp)
            end_index = self.df.index.get_loc(seq_obj.end_timestamp)

            expected_seq = self.df.iloc[start_index:end_index + 1].values.tolist()
            self.assertTrue((seq_obj.sequence_data, expected_seq))

            # Verify the column dictionary
            for feature_name, index in sequence_set.feature_dict.items():
                # Extract the slice of the 2D array corresponding to this feature
                sequence_slice = [row[index] for row in seq_obj.sequence_data]

                # Compare it to the original DataFrame column for this feature
                original_values = self.df[feature_name].iloc[start_index:end_index + 1].tolist()
                self.assertTrue(compare_lists_with_nan(sequence_slice, original_values))

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_retrieve_sequence_slice(self, mock_get_df_data):
        mock_get_df_data.return_value = self.df
        # Randomly select 3 features from the DataFrame
        sample_columns = random.sample(list(self.df.columns), 3)
        sequence_length = 10

        StockSequenceSetService.create_sequence_set(sequence_length = sequence_length, dataset_type = "stock", ticker = self.ticker, interval = self.interval)

        self.assertTrue(Sequence.objects.exists())

        results = StockSequenceSetService.retrieve_sequence_slice(sequence_length = sequence_length, feature_list = sample_columns, start_date = self.start_date, end_date = self.end_date,ticker = self.ticker, interval = self.interval, dataset_type = "stock")

        self.df.index = pd.to_datetime(self.df.index)

        for record in results:
            # get new arr from df
            df_slice = self.df.loc[record['start_timestamp']:record['end_timestamp']]
            new_arr = df_slice[sample_columns].values

            # Assert that the data matches
            self.assertTrue(np.array_equal(new_arr, record['sliced_data'], equal_nan=True))

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_add_feature_row(self, mock_get_df_data):
        mock_get_df_data.return_value = self.df
        sequence_length = 10
        StockSequenceSetService.create_sequence_set(sequence_length = sequence_length, dataset_type = "stock", ticker = self.ticker, interval = self.interval)


        X_new = np.random.rand(len(Sequence.objects.all()), 10, 10)

        # add nan val to last part of sequence
        X_new[:, -2:, :] = np.nan

        SequenceSetService.add_feature_row(SequenceSet.objects.first(), X_new)

        sequences = []
        for sequence in Sequence.objects.all().order_by('start_timestamp').reverse():
            sequence_data_with_nan = [[np.nan if val is None else val for val in row] for row in sequence.sequence_data]
            sequences.append(SequenceService.transpose(sequence_data_with_nan))

        arr_3d = np.array(sequences)
        self.assertTrue(np.allclose(arr_3d[:,:,-10:], X_new, equal_nan=True))

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_add_new_features(self, mock_get_df_data):
        mock_get_df_data.return_value = self.df
        sequence_length = 10
        StockSequenceSetService.create_sequence_set(sequence_length = sequence_length, dataset_type = "stock", ticker = self.ticker, interval = self.interval)

        new_features = ['new_feature1', 'new_feature2', 'y', 'close-1']
        self.df[new_features] = np.random.rand(len(self.df), 4)

        DatasetManagerService.update_existing_stock_data(deepcopy(self.df), self.ticker, self.interval)
        FeatureTrackerService.update_feature_tracker()

        mock_get_df_data.return_value = self.df
        StockSequenceSetService.add_new_features(sequence_set = SequenceSet.objects.first(), ticker = self.ticker, interval = self.interval, dataset_type = "stock")

        feature_dict = SequenceSet.objects.first().feature_dict

        self.assertTrue(sorted(list(range(len(self.df.columns)))) == sorted(list(feature_dict.values())))
        self.assertTrue(all([feature in feature_dict for feature in new_features]))

        for sequence in Sequence.objects.all().order_by('start_timestamp').reverse():
            df_seq = self.df.loc[sequence.start_timestamp:sequence.end_timestamp]
            sequence.sequence_data = np.array(SequenceService.transpose(sequence.sequence_data))
            for feature in self.df.columns:
                index = feature_dict[feature]
                self.assertTrue(compare_lists_with_nan(sequence.sequence_data[:,index], np.array(df_seq[feature].values.tolist())))





def compare_lists_with_nan(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


    