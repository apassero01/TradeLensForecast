from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
import json
import pandas as pd
import numpy as np
from setuptools.dist import sequence

from dataset_manager.models import DataSet
from dataset_manager.services import DataSetService
from sequenceset_manager.services import SequenceSetService, StockSequenceSetService, SequenceService
import math
from sequenceset_manager.models import SequenceSet, Sequence
from datetime import datetime, timedelta
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

        dataset = DataSet.objects.first()
        DataSetService.update_existing_dataset(deepcopy(self.df), dataset)

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


    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_update_recent_no_new_data(self, mock_get_df_data):
        """
        Test that no updates are made when there is no new data.
        """
        mock_get_df_data.return_value = self.df

        # Create a real sequence set (with sequences already in the DB)
        sequence_length = 10
        sequence_set = StockSequenceSetService.create_sequence_set(
            sequence_length=sequence_length,
            dataset_type='stock',
            ticker=self.ticker,
            interval=self.interval
        )

        # Call update_recent with the same DataFrame (no new rows)
        StockSequenceSetService.update_recent(
            sequence_set,
            ticker=self.ticker,
            interval=self.interval,
            dataset_type='stock'
        )

        # Verify that no new sequences were created
        self.assertEqual(
            Sequence.objects.count(),
            len(self.df) - sequence_length + 1
        )

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_update_recent_with_new_data(self, mock_get_df_data):
        """
        Test that new sequences are created when new data is added.
        """
        # First, mock initial data and create the set
        mock_get_df_data.return_value = self.df
        sequence_length = 10
        sequence_set = StockSequenceSetService.create_sequence_set(
            sequence_length=sequence_length,
            dataset_type='stock',
            ticker=self.ticker,
            interval=self.interval
        )

        # Simulate new data being added to the DataFrame
        new_data = {
            '2021-01-02 00:00:00+00:00': [100, 200, 300, 400],
            '2021-01-03 00:00:00+00:00': [110, 210, 310, 410]
        }
        new_df = self.df._append(pd.DataFrame(new_data).T)  # Use _append for newer pandas
        new_df.index = pd.to_datetime(new_df.index, format='%Y-%m-%d %H:%M:%S%z')
        mock_get_df_data.return_value = new_df

        # Call update_recent with the updated DataFrame
        StockSequenceSetService.update_recent(
            sequence_set,
            ticker=self.ticker,
            interval=self.interval,
            dataset_type='stock'
        )

        # Verify that new sequences were created
        expected_num_sequences = len(new_df) - sequence_length + 1
        self.assertEqual(Sequence.objects.count(), expected_num_sequences)

        # Verify that the last sequence includes the new data
        last_sequence = Sequence.objects.order_by('-end_timestamp').first()
        self.assertEqual(
            last_sequence.end_timestamp,
            pd.to_datetime('2021-01-03 00:00:00+00:00')
        )

    @patch('sequenceset_manager.services.StockSequenceSetService.get_df_data')
    def test_update_recent_timesteps_exist(self, mock_get_df_data):
        """
        After we append new rows and call update_recent, verify that for each
        row i in the final DataFrame (up to len(df)-sequence_length),
        there is exactly one Sequence whose start_timestamp and end_timestamp
        match the DataFrame index (accounting for type/timezone differences).
        """
        # 1) Create the initial sequence set
        mock_get_df_data.return_value = self.df
        sequence_length = 10
        sequence_set = StockSequenceSetService.create_sequence_set(
            sequence_length=sequence_length,
            dataset_type='stock',
            ticker=self.ticker,
            interval=self.interval
        )

        # 2) Create new data that extends the dataset
        new_data = {
            '2019-01-02 00:00:00+00:00': [100, 200, 300, 400],
            '2019-01-03 00:00:00+00:00': [110, 210, 310, 410],
            '2019-01-04 00:00:00+00:00': [120, 220, 320, 420],
        }
        new_df = self.df._append(pd.DataFrame(new_data).T)
        new_df.index = pd.to_datetime(new_df.index)  # Pandas Timestamp index
        mock_get_df_data.return_value = new_df

        # 3) Call update_recent
        StockSequenceSetService.update_recent(
            sequence_set,
            ticker=self.ticker,
            interval=self.interval,
            dataset_type='stock'
        )

        # 4) Sort the final DataFrame by date
        final_df = new_df.sort_index()

        # 5) For each valid "start row" i in final_df, ensure exactly one matching Sequence
        for i in range(len(final_df)-1, len(final_df) - sequence_length + 1, -1):
            # Pandas Timestamps
            start_ts_pd = final_df.index[i - sequence_length + 1]

            # Convert to naive Python datetimes (if DB is naive). If DB is TZ-aware (UTC),
            # replace tzinfo=None with tzinfo=timezone.utc or unify as needed.
            start_ts_dt = start_ts_pd.to_pydatetime().replace(tzinfo=None)
            # 5a) Find any Sequence in the DB with that start_timestamp
            matches = Sequence.objects.filter(
                sequence_set=sequence_set,
                start_timestamp=start_ts_dt
            )
            self.assertEqual(
                matches.count(),
                1,
                msg=(
                    f"Expected exactly one Sequence starting at {start_ts_dt}, "
                    f"found {matches.count()}."
                )
            )

            # 5b) Check that its end_timestamp matches the expected
            seq_obj = matches.first()
            db_start = seq_obj.start_timestamp.replace(tzinfo=None)
            self.assertEqual(
                db_start,
                start_ts_dt,
                msg=(
                    f"Sequence at {start_ts_dt} has incorrect end_timestamp. "
                    f"Expected {db_start}, got {start_ts_dt}."
                )
            )

class GetSequencesFunctionTest(TestCase):
    def setUp(self):
        """
        Create multiple SequenceSets and Sequences to test 'get_sequences'.
        We'll also set up some random data to simulate multiple features.
        """
        self.num_features = 4
        self.feature_names = [f"feat_{i}" for i in range(self.num_features)]

        # Create two SequenceSets for demonstration
        self.sequence_set_1 = SequenceSet.objects.create(
            dataset_type='test',
            sequence_length=5,
            start_timestamp=datetime(2021, 1, 1),
            end_timestamp=datetime(2021, 1, 31),
            feature_dict={feature: idx for idx, feature in enumerate(self.feature_names)},
            metadata={'test_meta': 'set1'}
        )
        self.sequence_set_2 = SequenceSet.objects.create(
            dataset_type='test',
            sequence_length=5,
            start_timestamp=datetime(2021, 2, 1),
            end_timestamp=datetime(2021, 2, 28),
            feature_dict={feature: idx for idx, feature in enumerate(self.feature_names)},
            metadata={'test_meta': 'set2'}
        )

        # Generate some fake data arrays and create Sequences for each set
        # For simplicity, each Sequence will be 2D: shape = (sequence_length, num_features)
        self.sequence_ids = range(0,5)
        base_date = datetime(2021, 1, 1)


        sequences_1 = []
        for i in range(3):
            # Create a random 5x4 matrix
            data = np.random.rand(5, self.num_features).tolist()
            seq_obj = Sequence(
                sequence_set=self.sequence_set_1,
                start_timestamp=base_date + timedelta(days=i),
                end_timestamp=base_date + timedelta(days=i + 4),
                sequence_length=5,
                sequence_data=data,
                pk=i
            )
            sequences_1.append(seq_obj)

        SequenceService.save_sequences(sequences_1)


        # Another set with different data
        base_date_2 = datetime(2021, 2, 1)
        sequences_2 = []
        for i in range(2):
            data = np.random.rand(5, self.num_features).tolist()
            seq_obj = Sequence(
                sequence_set=self.sequence_set_2,
                start_timestamp=base_date_2 + timedelta(days=i),
                end_timestamp=base_date_2 + timedelta(days=i + 4),
                sequence_length=5,
                sequence_data=data,
                pk= i + 3
            )
            sequences_2.append(seq_obj)

        SequenceService.save_sequences(sequences_2)



    def test_get_sequences_single_feature(self):
        """
        Test retrieving data for a single feature from multiple sequences
        across two SequenceSets. Check that the slicing is correct and
        the results maintain the order of the sequence_ids we provided.
        """
        # We'll only fetch the first feature, e.g. 'feat_0'
        feature_list = ['feat_0']
        results = SequenceService.get_sequences(
            sequence_ids=self.sequence_ids,
            feature_list=feature_list
        )

        # Check that 'results' is the same length as self.sequence_ids
        self.assertEqual(len(results), len(self.sequence_ids))

        # The order of results must match the order of self.sequence_ids
        for idx, seq_id in enumerate(self.sequence_ids):
            self.assertEqual(results[idx]['id'], seq_id)

            # Verify the shape of sliced_data is 5 (sequence_length) x 1 (feature_list size)
            sliced_data = results[idx]['sliced_data']
            self.assertEqual(len(sliced_data), 5)
            self.assertEqual(len(sliced_data[0]), 1)

    def test_get_sequences_multi_feature(self):
        """
        Test retrieving multiple features from the sequences.
        Check that we slice out 2 features correctly, and
        the shape is (sequence_length x 2).
        """
        feature_list = ['feat_0', 'feat_2']
        results = SequenceService.get_sequences(
            sequence_ids=self.sequence_ids,
            feature_list=feature_list
        )

        self.assertEqual(len(results), len(self.sequence_ids))

        for idx, seq_id in enumerate(self.sequence_ids):
            rec = results[idx]
            self.assertEqual(rec['id'], seq_id)

            # shape: (5, 2) -> 5 time steps, 2 features
            sliced_data = rec['sliced_data']
            self.assertEqual(len(sliced_data), 5)
            self.assertEqual(len(sliced_data[0]), 2)

    def test_get_sequences_empty_feature_list(self):
        """
        Test handling of an empty feature_list, which should yield an empty slice
        or array[]::double precision[] in SQL. The result is a shape of (5, 0).
        """
        feature_list = []
        results = SequenceService.get_sequences(
            sequence_ids=self.sequence_ids,
            feature_list=feature_list
        )

        for idx, seq_id in enumerate(self.sequence_ids):
            rec = results[idx]
            self.assertEqual(rec['id'], seq_id)
            sliced_data = rec['sliced_data']

            # shape: (5, 0) -> each row has 0 columns
            self.assertEqual(len(sliced_data), 0)
            for row in sliced_data:
                self.assertEqual(len(row), 0)

    def test_get_sequences_non_existent_id(self):
        """
        Test that a non-existent Sequence ID triggers a ValidationError.
        """
        bad_id = max(self.sequence_ids) + 9999
        with self.assertRaisesMessage(
                Exception,
                f"Sequence with ID {bad_id} does not exist."
        ):
            SequenceService.get_sequences(sequence_ids=[bad_id], feature_list=['feat_0'])

    def test_get_sequences_missing_feature(self):
        """
        Test that a missing feature in the feature_dict triggers a ValidationError.
        """
        with self.assertRaisesMessage(
                Exception,
                "Feature 'feat_999' not found"
        ):
            SequenceService.get_sequences(sequence_ids=self.sequence_ids, feature_list=['feat_999'])




def compare_lists_with_nan(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


    