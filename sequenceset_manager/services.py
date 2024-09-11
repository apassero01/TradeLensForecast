import json
import math
from abc import ABC, abstractmethod

from django.utils.timezone import override

from sequenceset_manager.models import SequenceSet, Sequence, \
    FeatureSequence
from django.db import transaction
# import requests
# import pandas as pd
# from django.db.models.expressions import RawSQL
# from django.db.models import Case, When, Value, F, FloatField, Func
# from django.db import models
# from django.core.exceptions import ValidationError
# from django.db import connection
import numpy as np


# class SequencesetManagerService:
#
#     # def update_recent_sequences(ticker, interval, sequence_length):
#     #     '''
#     #     Update the most recent sequences for a given stock ticker
#     #     '''
#     #     # get most recent sequence
#     #     most_recent_seq = StockSequence.objects.filter(ticker=ticker, timeframe=interval).order_by('-end_timestamp').first()
#     #     if most_recent_seq is None:
#     #         print(f"No sequences found for {ticker}")
#     #         return None
#
#     #     # get most recent date
#     #     most_recent_date = most_recent_seq.start_timestamp
#
#     @staticmethod
#     def refresh_features():
#         '''
#         Refresh the feature dictionary for a given stock ticker
#         '''
#
#         try:
#             response = requests.get(f"http://localhost:8000/dataset_manager/get_stock_features/")
#             true_list = response.json()
#         except requests.exceptions.RequestException as e:
#             print(e)
#
#         feature_dict = FeatureDict.objects.first()
#         feature_dict_list = list(feature_dict.feature_dict.keys())
#
#         if sorted(true_list) == sorted(feature_dict_list):
#             print("No new features to add")
#             return
#
#         new_features = list(set(true_list) - set(feature_dict_list))
#
#         sequenceSetTrackers = SequenceSetTracker.objects.all()
#
#         sorted_cols = sorted(new_features)
#         indices_seq = list(range(len(new_features)))
#         start_value = max(feature_dict.feature_dict.values()) + 1
#
#         indices_seq = [start_value + i for i in indices_seq]
#
#         with transaction.atomic():
#             for sequenceSetTracker in sequenceSetTrackers:
#                 df = SequencesetManagerService.get_stock_dataset(sequenceSetTracker.ticker, sequenceSetTracker.timeframe,
#                                                                  sequenceSetTracker.start_timestamp, sequenceSetTracker.end_timestamp)
#
#                 df.fillna(-999)
#                 df_new_features = df[sorted_cols]
#                 df_values = df_new_features.values
#                 timestamps = df_new_features.index
#
#                 new_feature_dict = {col: index for col, index in zip(sorted_cols, indices_seq)}
#
#                 sequences = []
#                 for i in range(len(df_new_features) - 1, -1, -1):
#                     start_index = i - sequenceSetTracker.sequence_length + 1
#                     if start_index < 0:
#                         break
#
#                     end_idx = i
#
#                     seq = df_values[start_index:end_idx + 1, :]
#                     start_date = timestamps[start_index]
#                     end_date = timestamps[end_idx]
#
#                     sequences.append(seq)
#
#                 SequencesetManagerService.add_feature_row(sequenceSetTracker, sequences)
#
#             feature_dict.feature_dict.update(new_feature_dict)
#             feature_dict.save()
#
#
#     @staticmethod
#     def add_feature_row(sequenceSetTracker, new_features):
#         seq_objs = StockSequence.objects.filter(ticker=sequenceSetTracker.ticker, timeframe=sequenceSetTracker.timeframe,
#                                                 sequence_length=sequenceSetTracker.sequence_length).order_by('start_timestamp').reverse()
#         if len(seq_objs) != len(new_features):
#             raise ValidationError("Length of new features does not match the length of the sequence objects")
#
#         with transaction.atomic():
#             for index, seq_obj in enumerate(seq_objs):
#                 obj_id = seq_obj.id
#                 seq_data = new_features[index]
#
#                 seq_data = [[None if math.isnan(val) else val for val in row] for row in seq_data]
#
#                 seq_data = SequencesetManagerService.transpose(seq_data)
#
#                 # Construct the SQL string, ensuring 'None' is replaced with 'NULL'
#                 array2d_str = ','.join([
#                     f"ARRAY[{','.join('NULL' if x is None else str(x) for x in row)}]::float8[]" for row in seq_data
#                 ])
#                 StockSequence.objects.filter(pk=obj_id).update(
#                     sequence_data=RawSQL(f"\"sequence_data\" || ARRAY[{array2d_str}]::float8[]", [])
#                 )
#
#
#     @staticmethod
#     def create_new_sequence_set(ticker, interval, sequence_length, start_date=None, end_date=None):
#         '''
#         Create a new sequence set for a given stock ticker
#         '''
#         ticker = ticker.upper()
#         # Check if the sequence set already exists
#         existing_sequences = StockSequence.objects.filter(ticker=ticker, timeframe=interval, sequence_length=sequence_length)
#         if existing_sequences.exists():
#             print(f"Sequence set for {ticker} already exists")
#             return
#         df = SequencesetManagerService.get_stock_dataset(ticker, interval, start_date, end_date)
#         if df is None:
#             print(f"Error fetching stock data for {ticker}")
#             return None
#         print("Creating Sequence objects")
#         stock_sequences, feature_dict = SequencesetManagerService.create_sequence_objects(ticker, interval,
#                                                                                           sequence_length, df)
#         print("Saving Sequence objects")
#
#         with transaction.atomic():
#             SequencesetManagerService.save_stock_sequences(stock_sequences, feature_dict)
#
#             if start_date is None:
#                 start_date = df.index[0]
#             if end_date is None:
#                 end_date = df.index[-1]
#
#             tracker = SequenceSetTrackerService.create_sequence_set_tracker(ticker, interval, sequence_length, start_date, end_date)
#
#         return stock_sequences
#
#
#     @staticmethod
#     def create_sequence_objects(ticker, timeframe, sequence_length, df):
#         '''
#         Create a 3D sequence from a DataFrame
#         '''
#         ticker = ticker.upper()
#
#         df_cols = df.columns
#         df_cols = sorted(df_cols)
#         df = df[df_cols]
#
#         timestamps = df.index
#         df_values = df.values
#
#         indices_seq = list(range(len(df_cols)))
#
#         if FeatureDict.objects.first() is not None:
#             feature_dict = FeatureDict.objects.first()
#         else:
#             feature_dict = {col: index for col, index in zip(df_cols, indices_seq)}
#
#         stock_sequences = []
#
#         for i in range(len(df_values) - 1, -1, -1):
#             start_index = i - sequence_length + 1
#             if start_index < 0:
#                 break
#
#             end_idx = i
#
#             seq = df_values[start_index:end_idx + 1, :]
#             start_date = timestamps[start_index]
#             end_date = timestamps[end_idx]
#
#             stock_sequences.append(StockSequence(
#                 ticker=ticker,
#                 start_timestamp=start_date,
#                 end_timestamp=end_date,
#                 sequence_length=sequence_length,
#                 sequence_data=seq.tolist(),
#                 timeframe=timeframe,
#             ))
#
#         feature_dict_obj = FeatureDict(
#             ticker=ticker,
#             feature_dict=feature_dict,
#             timeframe=timeframe
#         )
#
#         return stock_sequences, feature_dict_obj
#
#     @staticmethod
#     def save_stock_sequences(stock_sequences, feature_dict):
#         '''
#         Save a list of StockSequence objects to the database
#         '''
#         print("Transposing Sequences")
#         for seq in stock_sequences:
#             seq.sequence_data = SequencesetManagerService.transpose(seq.sequence_data)
#         print("Saving Sequences")
#         batch_size = 1000
#         with transaction.atomic():
#             for i in range(0, len(stock_sequences), batch_size):
#                 StockSequence.objects.bulk_create(stock_sequences[i:i + batch_size])
#             if FeatureDict.objects.first() is None:
#                 feature_dict.save()
#
#     @staticmethod
#     def get_stock_dataset(ticker, interval="1d", start_date=None, end_date=None):
#         '''
#         retreive stock dataset from dataset_manager
#         '''
#
#         params = {
#             "interval": interval,
#             "start_date": start_date,
#             "end_date": end_date
#         }
#         try:
#             response = requests.get(f"http://localhost:8000/dataset_manager/get_stock_data/{ticker}", params=params)
#             data = response.json()
#             return pd.DataFrame(data).T
#         except requests.exceptions.RequestException as e:
#             print(e)
#             return None
#
#     @staticmethod
#     def retrieve_sequence_slice(ticker, features, interval='1d', start_date=None, end_date=None, sequence_length=None):
#         queryset = StockSequence.objects.filter(ticker=ticker)
#         print(queryset.count())
#
#         ## TODO Fix what I am about to write
#         column_dict = FeatureDict.objects.first().feature_dict
#         indices = [column_dict[feature] for feature in features]
#
#         print(interval)
#         queryset = queryset.filter(timeframe=interval)
#         print(queryset.count())
#
#         if start_date:
#             queryset = queryset.filter(start_timestamp__gte=start_date)
#         if end_date:
#             queryset = queryset.filter(end_timestamp__lte=end_date)
#         if sequence_length:
#             queryset = queryset.filter(sequence_length=sequence_length)
#
#         print(queryset.count())
#         # Prepare the SQL for selecting specific indices
#         queryset = queryset.annotate(
#             sliced_data=RawSQL(
#                 f"ARRAY[{', '.join([f'sequence_data[{i + 1}:{i + 1}]' for i in indices])}]",
#                 []
#             )
#         ).values('id', 'ticker', 'start_timestamp', 'end_timestamp', 'sequence_length', 'sliced_data')
#
#         print(queryset.count())
#         result = list(queryset.order_by('start_timestamp'))
#
#         # Flatten and transpose the array
#         for record in result:
#             # Replace None with NaN
#             record['sliced_data'] = [
#                 [np.nan if val is None else val for val in row]
#                 for row in record['sliced_data']
#             ]
#
#             # Convert to a NumPy array, squeeze it, transpose, and convert back to a list
#             record['sliced_data'] = np.array(record['sliced_data'])
#             record['sliced_data'] = np.squeeze(record['sliced_data']).T.tolist()
#
#         return result
#
#     @staticmethod
#     def transpose(matrix_list):
#         '''
#         Transpose a list of matrices
#         '''
#         return [list(row) for row in zip(*matrix_list)]
#
#
#
#
# class FeatureDictService:
#
#     @staticmethod
#     def get_feature_dict():
#         '''
#         Get the feature dictionary
#         '''
#         feature_dict = FeatureDict.objects.first()
#         if feature_dict is None:
#             raise ValidationError("Feature dictionary not found")
#         return feature_dict.feature_dict
#
#     @staticmethod
#     def generate_new_feature_dict(df_cols):
#         indices_seq = list(range(len(df_cols)))
#
#         feature_dict = {col: index for col, index in zip(df_cols, indices_seq)}
#
#         return feature_dict
#
#     @staticmethod
#     def add_new_feature(new_features):
#         current_feature_dict = FeatureDict.objects.first()
#
#         if current_feature_dict is None:
#             raise ValidationError("Feature dictionary not found")
#
#         dict = current_feature_dict.feature_dict
#
#         max_index = max(dict.values())
#
#         new_feature_dict = {feature: max_index + i + 1 for i, feature in enumerate(new_features)}
#
#         current_feature_dict.feature_dict.update(new_feature_dict)
#
#         current_feature_dict.save()
#
#         return current_feature_dict.feature_dict
#
#
#
# class SequenceSetTrackerService:
#
#     @staticmethod
#     def get_sequence_set_tracker(ticker, interval, sequence_length):
#         '''
#         Get the SequenceSetTracker object for a given stock ticker
#         '''
#         sequence_set_tracker = SequenceSetTracker.objects.filter(ticker=ticker, timeframe=interval,
#                                                                  sequence_length=sequence_length).first()
#         return sequence_set_tracker
#
#     @staticmethod
#     def create_sequence_set_tracker(ticker, interval, sequence_length, start_date, end_date):
#         '''
#         Create a new SequenceSetTracker object
#         '''
#         sequence_set_tracker = SequenceSetTracker(
#             ticker=ticker,
#             timeframe=interval,
#             sequence_length=sequence_length,
#             start_timestamp=start_date,
#             end_timestamp=end_date
#         )
#         sequence_set_tracker.save()
#         return sequence_set_tracker
#
#     @staticmethod
#     def update_sequence_set_tracker(ticker, interval, sequence_length, start_date, end_date):
#         '''
#         Update the SequenceSetTracker object
#         '''
#         sequence_set_tracker = SequenceSetTracker.objects.filter(ticker=ticker, timeframe=interval,
#                                                                  sequence_length=sequence_length).first()
#         if sequence_set_tracker is None:
#             return None
#         sequence_set_tracker.start_date = start_date
#         sequence_set_tracker.end_date = end_date
#         sequence_set_tracker.save()
#         return sequence_set_tracker



class SequenceSetService(ABC):

    @staticmethod
    def get_df_data(**kwargs):
        pass

    @staticmethod
    def get_meta_data(**kwargs):
        pass

    @classmethod
    def create_sequence_set(cls, sequence_length, **kwargs):
        '''
        Create a new sequence set
        '''
        df = cls.get_df_data(**kwargs)

        if df is None:
            print("Error fetching data")
            return None

        feature_names = list(df.columns)
        start_timestamp = df.index[0]
        end_timestamp = df.index[-1]

        sequence_set = SequenceSet(
            dataset_type=kwargs.get('dataset_type'),
            sequence_length=sequence_length,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            metadata=kwargs,
            feature_names=feature_names
        )

        SequenceSetService.create_sequences(df, sequence_set)

        return sequence_set

    @staticmethod
    def create_sequences(df, sequence_set):
        '''
        Create sequences from a DataFrame
        '''
        df_cols = df.columns
        df_cols = sorted(df_cols)
        df = df[df_cols]

        df_values = df.values
        timestamps = df.index

        indices_seq = list(range(len(df.columns)))
        feature_dict = {col: index for col, index in zip(df.columns, indices_seq)}

        with transaction.atomic():
            sequence_set.save()
            for i in range(len(df_values) - 1, -1, -1):
                start_index = i - sequence_set.sequence_length + 1
                if start_index < 0:
                    break

                end_idx = i

                seq = df_values[start_index:end_idx + 1, :]
                start_date = timestamps[start_index]
                end_date = timestamps[end_idx]

                sequence = Sequence(
                    sequence_set=sequence_set,
                    start_timestamp=start_date,
                    end_timestamp=end_date,
                    feature_names=df_cols,
                )
                sequence.save()

                features_sequence = []
                for feature in df_cols:
                    feature_index = feature_dict[feature]
                    values = seq[:, feature_index]
                    feature_sequence = FeatureSequence(
                        sequence=sequence,
                        feature_name=feature,
                        values=values.tolist()
                    )
                    features_sequence.append(feature_sequence)

                FeatureSequence.objects.bulk_create(features_sequence)

    @staticmethod
    def get_feature_slice(sequence_length, feature_list, start_date = None, end_date = None, **kwargs):
        '''
        Get sequence data for a given query
        '''

        query = SequenceSet.objects.filter(dataset_type=kwargs.get("dataset_type"), metadata__contains=kwargs, sequence_length=sequence_length)
        print(len(query))

        filled_sequences = []
        for sequence_set in query:
            print("GOING OVER FIRST SEQ SET ")
            sequences = Sequence.objects.filter(sequence_set=sequence_set)

            if start_date:
                sequences = sequences.filter(start_timestamp__gte=start_date)
            if end_date:
                sequences = sequences.filter(end_timestamp__lte=end_date)
            print("GOING OVER SEQUENCES")
            for seq in sequences:
                feature_arr = np.zeros((sequence_length, len(feature_list)))
                for index, feature in enumerate(feature_list):
                    print("ADDING FEATURE " + str(index))
                    feature_sequence = FeatureSequence.objects.get(sequence=seq, feature_name=feature)
                    feature_arr[:, index] = list(feature_sequence.values)

                seq.feature_data = feature_arr
                filled_sequences.append(seq)

        return filled_sequences

    @staticmethod
    def update_features(sequence_set):
        '''
        Update the feature dictionary for a given sequence set
        '''

        df = SequenceSetService.get_df_data()

        cur_feature_names = sequence_set.feature_names
        new_feature_names = df.columns
        new_features = sorted(list(set(new_feature_names) - set(cur_feature_names)))

        if len(new_features) == 0:
            print("No new features to add")
            return

        df_new_features = df[new_features]

        df_values = df_new_features.values
        timestamps = df_new_features.index

        indices_seq = list(range(len(new_features)))
        feature_dict = {col: index for col, index in zip(new_features, indices_seq)}

        with transaction.atomic():
            for i in range(len(df_new_features) - 1, -1, -1):
                start_index = i - sequence_set.sequence_length + 1
                if start_index < 0:
                    break

                end_idx = i

                new_seq = df_values[start_index:end_idx + 1, :]
                start_date = timestamps[start_index]
                end_date = timestamps[end_idx]

                sequence_obj = Sequence.objects.get(sequence_set=sequence_set, start_timestamp=start_date, end_timestamp=end_date)

                if sequence_obj is None:
                    print(f"Sequence not found for {start_date} - {end_date}")
                    continue

                feature_sequences = []

                for feature in df_new_features.columns:
                    feature_index = feature_dict[feature]
                    values = new_seq[:, feature_index]
                    feature_sequence = FeatureSequence(
                        sequence=sequence_obj,
                        feature_name=feature,
                        values=values.tolist()
                    )
                    feature_sequences.append(feature_sequence)

                FeatureSequence.objects.bulk_create(feature_sequences)

            sequence_set.feature_names.extend(new_features)


class StockSequenceSetService(SequenceSetService):

    @staticmethod
    def get_df_data(**kwargs):
        '''
        Get stock data for a given stock ticker
        '''
        ticker = kwargs.get('ticker')
        interval = kwargs.get('interval')
        start_date = kwargs.get('start_date', None)
        end_date = kwargs.get('end_date', None)

        print(ticker)
        print(interval)
        print(start_date)
        print(end_date)

        df = SequencesetManagerService.get_stock_dataset(ticker, interval, start_date, end_date)
        return df















