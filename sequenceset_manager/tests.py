from unittest.mock import patch, MagicMock

from django.test import TestCase, Client
import json
import pandas as pd
import numpy as np
from zmq.backend import first

from dataset_manager.models import DataSetTracker
from dataset_manager.services import DatasetManagerService, DatasetTrackerService, FeatureTrackerService
from sequenceset_manager.services import SequencesetManagerService, SequenceSetTrackerService
import math
from sequenceset_manager.models import StockSequence, FeatureDict
from datetime import datetime
import random
from copy import deepcopy

class SequencesetManagerServiceTest(TestCase):
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
        sequence_length = 5
        timeframe = self.interval
        
        sequences, feature_dict = SequencesetManagerService.create_sequence_objects(self.ticker, timeframe, sequence_length, self.df)

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

            columns_sorted = sorted(self.df.columns)
            expected_seq = self.df[columns_sorted].iloc[start_index:end_index + 1].values.tolist()
            # print(seq_obj.sequence_data)
            # print("\n")
            # print(expected_seq)

            self.assertTrue((seq_obj.sequence_data, expected_seq))

            # print(self.df['close'].tail(10))
            # col = [row[seq_obj.column_dict['close']] for row in seq_obj.sequence_data]
            # print(col)

            # Verify the column dictionary
            for feature_name, index in feature_dict.feature_dict.items():
                # Extract the slice of the 2D array corresponding to this feature
                sequence_slice = [row[index] for row in seq_obj.sequence_data]

                # Compare it to the original DataFrame column for this feature
                original_values = self.df[feature_name].iloc[start_index:end_index + 1].tolist()
                self.assertTrue(compare_lists_with_nan(sequence_slice, original_values))

            # Ensure the ticker and timeframe match
            self.assertEqual(seq_obj.ticker, self.ticker.upper())
            self.assertEqual(seq_obj.timeframe, timeframe)
        
    def test_save_stock_sequences(self):
        sequence_length = 5
        timeframe = self.interval
        
        sequences, feature_dict = SequencesetManagerService.create_sequence_objects(self.ticker, timeframe, sequence_length, self.df)
        SequencesetManagerService.save_stock_sequences(deepcopy(sequences), feature_dict)

        # Verify that the sequences were saved to the database
        saved_sequences = StockSequence.objects.filter(ticker=self.ticker.upper(), timeframe=timeframe)
        self.assertEqual(len(saved_sequences), len(sequences))

        # Verify that the saved sequences match the original sequences
        for saved_seq, orig_seq in zip(saved_sequences, sequences):
            self.assertEqual(saved_seq.ticker, orig_seq.ticker)
            self.assertEqual(saved_seq.timeframe, orig_seq.timeframe)
            start_string = saved_seq.start_timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
            start_string = start_string[:-2] + ':' + start_string[-2:]
            self.assertEqual(start_string, orig_seq.start_timestamp)
            end_string = saved_seq.end_timestamp.strftime('%Y-%m-%d %H:%M:%S%z')
            end_string = end_string[:-2] + ':' + end_string[-2:]
            self.assertEqual(end_string, orig_seq.end_timestamp)
            self.assertEqual(saved_seq.sequence_length, orig_seq.sequence_length)

            saved_seq.sequence_data = SequencesetManagerService.transpose(saved_seq.sequence_data)
            for saved_row, orig_row in zip(saved_seq.sequence_data, orig_seq.sequence_data):
                self.assertTrue(compare_lists_with_nan(saved_row, orig_row))
        
        dict_queried = FeatureDict.objects.get(ticker=self.ticker.upper(), timeframe=timeframe).feature_dict
        self.assertEqual(dict_queried, feature_dict.feature_dict)

        # Clean up the saved sequences
        saved_sequences.delete()
    
    def test_retrieve_sequence_slice(self):
        # Randomly select 3 features from the DataFrame
        sample_columns = random.sample(list(self.df.columns), 3)
        print(sample_columns)
        sequence_length = 10  # Assume a sequence length for testing

        # Create StockSequence objects using the DataFrame
        stock_sequences, feature_dict = SequencesetManagerService.create_sequence_objects(
            ticker=self.ticker,
            timeframe=self.interval,
            sequence_length=sequence_length,
            df=self.df
        )

        # Save the created StockSequence objects to the database
        SequencesetManagerService.save_stock_sequences(stock_sequences, feature_dict)
       
        sequence_data = stock_sequences[0].sequence_data
        rows = len(sequence_data)  # Number of rows
        cols = len(sequence_data[0]) if rows > 0 else 0  # Number of columns

        print(f"Shape of the 2D list: {rows} x {cols}")
        for seq in stock_sequences:
            seq.sequence_data = SequencesetManagerService.transpose(seq.sequence_data)
        sequence_data = stock_sequences[0].sequence_data
        rows = len(sequence_data)  # Number of rows
        cols = len(sequence_data[0]) if rows > 0 else 0  # Number of columns

        print(f"Shape of the 2D list: {rows} x {cols}")

        # Retrieve the sequence slice using the retrieve_sequence_slice function
        start_date = self.start_date
        end_date = self.end_date

        result = SequencesetManagerService.retrieve_sequence_slice(
            ticker=self.ticker,
            interval=self.interval,
            features=sample_columns,
            start_date=start_date,
            end_date=end_date,
            sequence_length=sequence_length
        )
        print(len(result))

        feature_dict = FeatureDict.objects.get(ticker=self.ticker.upper(), timeframe=self.interval).feature_dict
        column_indices = [feature_dict[feature] for feature in sample_columns]

        # Compare the retrieved objects with the original stock_sequences
        for record in result:
            # Find the corresponding StockSequence object
            stock_sequence = next(
                seq for seq in stock_sequences 
                if seq.id == record["id"]
                and seq.id == record['id']
            )

            # Extract the relevant columns from the original StockSequence's sequence_data
            # column_indices = [stock_sequence.column_dict[feature] for feature in sample_columns]
            
            stock_sequence.sequence_data = np.array(stock_sequence.sequence_data)
            new_arr = stock_sequence.sequence_data[:, column_indices]
            
            # Assert that the data matches
            self.assertTrue(np.array_equal(new_arr, record['sliced_data']))

    def test_add_feature_row(self):
        sequences, feature_dict = SequencesetManagerService.create_sequence_objects(self.ticker, self.interval, 10, self.df)
        SequencesetManagerService.save_stock_sequences(deepcopy(sequences), feature_dict)
        sequence_set_tracker = SequenceSetTrackerService.create_sequence_set_tracker(self.ticker, self.interval, 10, self.start_date,self.end_date)


        X_new = np.random.rand(len(sequences), 10, 10)

        # add nan val to last part of sequence
        X_new[:, -2:, :] = np.nan

        SequencesetManagerService.add_feature_row(sequence_set_tracker, X_new)

        sequences = []
        for stock_sequence in StockSequence.objects.filter(ticker=self.ticker.upper(), timeframe=self.interval):
            sequence_data_with_nan = [[np.nan if val is None else val for val in row] for row in stock_sequence.sequence_data]
            sequences.append(SequencesetManagerService.transpose(sequence_data_with_nan))

        arr_3d = np.array(sequences)
        self.assertTrue(np.allclose(arr_3d[:,:,-10:], X_new, equal_nan=True))

    @patch('sequenceset_manager.services.requests.get')
    @patch('sequenceset_manager.services.SequencesetManagerService.get_stock_dataset')
    def test_refresh_features(self, mock_get_stockdataset, mock_get):
        sequences, feature_dict = SequencesetManagerService.create_sequence_objects(self.ticker, self.interval, 5, self.df)
        SequencesetManagerService.save_stock_sequences(deepcopy(sequences), feature_dict)
        sequence_set_tracker = SequenceSetTrackerService.create_sequence_set_tracker(self.ticker, self.interval, 5, self.start_date, self.end_date)

        self.df.index = pd.to_datetime(self.df.index)

        new_features = ['new_feature1', 'new_feature2', 'y', 'close-1']
        self.df[new_features] = np.random.rand(len(self.df), 4)

        mock_get_stockdataset.return_value = self.df
        DatasetManagerService.update_existing_stock_data(self.df, self.ticker, self.interval)

        FeatureTrackerService.update_feature_tracker()

        true_list = FeatureTrackerService.get_feature_tracker().features

        mock_response = MagicMock()
        mock_response.json.return_value = true_list
        mock_get.return_value = mock_response


        SequencesetManagerService.refresh_features()


        feature_dict = FeatureDict.objects.first().feature_dict

        self.assertTrue(sorted(list(range(len(self.df.columns)))) == sorted(list(feature_dict.values())))

        self.assertTrue(all([feature in feature_dict for feature in new_features]))


        for stock_sequence in StockSequence.objects.filter(ticker=self.ticker.upper(), timeframe=self.interval):
            df_seq = self.df.loc[stock_sequence.start_timestamp:stock_sequence.end_timestamp]
            for feature in new_features:
                index = feature_dict[feature]
                self.assertTrue(compare_lists_with_nan(stock_sequence.sequence_data[index], df_seq[feature].values.tolist()))




def compare_lists_with_nan(list1, list2):
    if len(list1) != len(list2):
        return False
    for a, b in zip(list1, list2):
        if math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


    