import json
import math
from abc import ABC
from sequenceset_manager.models import SequenceSet, Sequence
from django.db import transaction
import requests
import pandas as pd
from django.db.models.expressions import RawSQL
from django.core.exceptions import ValidationError
import numpy as np

class SequenceSetService(ABC):
    @staticmethod

    def get_df_data():
        '''
        Get the data for a sequence set
        '''
        pass

    @staticmethod
    def retrieve_sequence_slice(sequence_length, feature_list, start_date = None, end_date = None, **kwargs):
        '''
        Retrieve a sequence slice
        '''

        queryset = SequenceSet.objects.filter(dataset_type = kwargs.get('dataset_type'), metadata__contains = kwargs, sequence_length = sequence_length)

        if len(queryset) > 1:
            raise ValidationError("Multiple sequence sets found")
        elif len(queryset) == 0:
            raise ValidationError("No sequence sets found")

        sequence_set = queryset.first()

        feature_dict = sequence_set.feature_dict
        indices = [feature_dict[feature] for feature in feature_list]

        queryset = Sequence.objects.filter(sequence_set = sequence_set)

        if start_date:
            queryset = queryset.filter(start_timestamp__gte = start_date)
        if end_date:
            queryset = queryset.filter(end_timestamp__lte = end_date)

        queryset = queryset.annotate(
            sliced_data=RawSQL(
                f"ARRAY[{', '.join([f'sequence_data[{i + 1}:{i + 1}]' for i in indices])}]",
                []
            )
        ).values('id', 'sequence_set', 'start_timestamp', 'end_timestamp', 'sequence_length', 'sliced_data')

        result = list(queryset.order_by('start_timestamp'))

        for record in result:
            record['sliced_data'] = [
                [np.nan if val is None else val for val in row]
                for row in record['sliced_data']
            ]

            record['sliced_data'] = np.array(record['sliced_data'])
            record['sliced_data'] = np.squeeze(record['sliced_data']).T.tolist()

        return result

    @classmethod
    def create_sequence_set(cls, sequence_length, **kwargs):
        '''
        Create a new sequence set
        '''
        df = cls.get_df_data(**kwargs)

        if df is None:
            print("Error fetching stock data")
            return None

        start_timestamp = df.index[0]
        end_timestamp = df.index[-1]

        sequence_set = SequenceSet(
            dataset_type=kwargs.get('dataset_type'),
            sequence_length=sequence_length,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            metadata=kwargs,
            feature_dict=SequenceSetService.create_feature_dict(df)
        )

        sequences = SequenceSetService.create_sequence_objects(sequence_set, df)
        SequenceSetService.save_sequence_set(sequence_set, sequences)

    @staticmethod
    def create_sequence_objects(sequence_set, df):
        '''
        Create a 3D sequence from a DataFrame
        '''
        feature_dict = sequence_set.feature_dict
        cols = list(feature_dict.keys())

        df = df[cols]
        df_values = df.values
        timestamps = df.index

        sequence_length = sequence_set.sequence_length

        stock_sequences = []
        for i in range(len(df_values) - 1, -1, -1):
            start_index = i - sequence_length + 1
            if start_index < 0:
                break

            end_idx = i

            seq = df_values[start_index:end_idx + 1, :]
            start_date = timestamps[start_index]
            end_date = timestamps[end_idx]

            stock_sequences.append(Sequence(
                sequence_set=sequence_set,
                start_timestamp=start_date,
                end_timestamp=end_date,
                sequence_length=sequence_length,
                sequence_data=seq.tolist(),
            ))

        return stock_sequences

    @classmethod
    def add_new_features(cls,sequence_set, **kwargs):
        '''
        Add new features to a sequence set
        '''
        df = cls.get_df_data(**kwargs)
        df.index = pd.to_datetime(df.index)

        feature_dict = sequence_set.feature_dict
        feature_dict_list = list(feature_dict.keys())

        if sorted(df.columns) == sorted(feature_dict_list):
            print("No new features to add")
            return

        new_features = list(set(df.columns) - set(feature_dict_list))

        sorted_cols = sorted(new_features)
        indices_seq = list(range(len(new_features)))
        start_value = max(feature_dict.values()) + 1

        indices_seq = [start_value + i for i in indices_seq]

        sequence_objs = Sequence.objects.filter(sequence_set=sequence_set).order_by('start_timestamp').reverse()

        with transaction.atomic():
            df_new_features = df[sorted_cols]
            df_values = df_new_features.values
            timestamps = df_new_features.index

            new_feature_dict = {col: index for col, index in zip(sorted_cols, indices_seq)}

            sequences = []
            sequences_adjusted = 0
            for i in range(len(df_new_features) - 1, -1, -1):

                start_index = i - sequence_set.sequence_length + 1
                if start_index < 0:
                    break

                end_idx = i

                seq = df_values[start_index:end_idx + 1, :]
                start_date = timestamps[start_index]
                end_date = timestamps[end_idx]

                current_sequence = sequence_objs[sequences_adjusted]
                cur_seq_start_timestamp = current_sequence.start_timestamp
                cur_seq_end_timestamp = current_sequence.end_timestamp

                if cur_seq_start_timestamp != start_date or cur_seq_end_timestamp != end_date:
                    raise ValidationError("Timestamps do not match")

                sequences.append(seq)

                sequences_adjusted += 1

            SequenceSetService.add_feature_row(sequence_set, sequences)
            sequence_set.feature_dict.update(new_feature_dict)
            sequence_set.save()


    @staticmethod
    def add_feature_row(sequence_set, new_features):
        seq_objs = Sequence.objects.filter(sequence_set=sequence_set).order_by('start_timestamp').reverse()
        if len(seq_objs) != len(new_features):
            raise ValidationError("Length of new features does not match the length of the sequence objects")

        with transaction.atomic():
            for index, seq_obj in enumerate(seq_objs):
                obj_id = seq_obj.id
                seq_data = new_features[index]

                seq_data = [[None if math.isnan(val) else val for val in row] for row in seq_data]

                seq_data = SequenceService.transpose(seq_data)

                # Construct the SQL string, ensuring 'None' is replaced with 'NULL'
                array2d_str = ','.join([
                    f"ARRAY[{','.join('NULL' if x is None else str(x) for x in row)}]::float8[]" for row in seq_data
                ])
                Sequence.objects.filter(pk=obj_id).update(
                    sequence_data=RawSQL(f"\"sequence_data\" || ARRAY[{array2d_str}]::float8[]", [])
                )

    @staticmethod
    def save_sequence_set(sequence_set, stock_sequences):
        '''
        Save a SequenceSet object and a list of Sequence objects to the database
        '''
        with transaction.atomic():
            sequence_set.save()
            SequenceService.save_sequences(stock_sequences)

    @staticmethod
    def create_feature_dict(df):
        '''
        Create a feature dictionary from a DataFrame
        '''
        df_cols = df.columns
        df_cols = sorted(df_cols)
        indices_seq = list(range(len(df_cols)))

        feature_dict = {col: index for col, index in zip(df_cols, indices_seq)}
        return feature_dict

class SequenceService(ABC):

    @staticmethod
    def save_sequences(sequences):
        '''
        Save a list of StockSequence objects to the database
        '''
        print("Transposing Sequences")
        for seq in sequences:
            seq.sequence_data = SequenceService.transpose(seq.sequence_data)
        print("Saving Sequences")
        batch_size = 1000
        with transaction.atomic():
            for i in range(0, len(sequences), batch_size):
                Sequence.objects.bulk_create(sequences[i:i + batch_size])

    @staticmethod
    def transpose(matrix_list):
        '''
        Transpose a list of matrices
        '''
        return [list(row) for row in zip(*matrix_list)]


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

        df = StockSequenceSetService.get_stock_dataset(ticker, interval, start_date, end_date)
        return df

    @staticmethod
    def get_stock_dataset(ticker, interval="1d", start_date=None, end_date=None):
        '''
        retreive stock dataset from dataset_manager
        '''

        params = {
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date
        }
        try:
            response = requests.get(f"http://localhost:8000/dataset_manager/get_stock_data/{ticker}", params=params)
            data = response.json()
            return pd.DataFrame(data).T
        except requests.exceptions.RequestException as e:
            print(e)
            return None






