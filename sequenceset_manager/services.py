import json
import math
from abc import ABC
from collections import defaultdict

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
        query = SequenceSet.objects.filter(dataset_type=kwargs.get('dataset_type'), metadata__contains=kwargs, sequence_length=sequence_length)
        if query.exists():
            print("Sequence set already exists")
            return

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
        cols = sorted(cols, key=lambda x: feature_dict[x])

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

    @staticmethod
    def get_all_sequence_set_metadata():
        '''
        Get all sequence set metadata
        '''
        metadata = []
        for sequence_set in SequenceSet.objects.all():
            curmeta_data = sequence_set.metadata
            curmeta_data['sequence_length'] = sequence_set.sequence_length
            curmeta_data['id'] = sequence_set.id
            metadata.append(sequence_set.metadata)

        return metadata



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

    @staticmethod
    def get_sequence_metadata_by_ids(sequence_ids):
        '''
        Get sequence metadata by ids
        '''
        print(sequence_ids)
        sequences = Sequence.objects.filter(id__in=sequence_ids)
        sequence_set = sequences.first().sequence_set
        metadata = []
        for sequence in sequences:
            meta_data = {}
            meta_data['id'] = sequence.id
            meta_data['start_timestamp'] = sequence.start_timestamp
            meta_data['end_timestamp'] = sequence.end_timestamp
            meta_data['sequence_length'] = sequence.sequence_length
            meta_data['metadata'] = sequence_set.metadata
            metadata.append(meta_data)

        return metadata

    @staticmethod
    def get_sequences(sequence_ids, feature_list=None, start_date=None, end_date=None):
        """
        Retrieve sliced sequence_data for each sequence_id in `sequence_ids`,
        grouped by their SequenceSet so we only fetch each set's feature_dict once.

        Returns a list of results in the same order as `sequence_ids`.
        Each entry in the returned list corresponds to one sequence ID,
        containing fields like:
          {
            'id': <sequence_id>,
            'sequence_set': <set_id>,
            'start_timestamp': ...,
            'end_timestamp': ...,
            'sequence_length': ...,
            'sliced_data': [...]
          }
        """
        if feature_list is None:
            feature_list = []

        if not sequence_ids:
            return []

        # 1) Keep track of the original position of each ID (so we can preserve order in the final output)
        id_to_index = {}
        for i, seq_id in enumerate(sequence_ids):
            id_to_index[seq_id] = i

        # 2) Group sequence IDs by sequence_set so we don't repeatedly load the same set
        set_groups = defaultdict(list)  # {set_id: [seq_id, seq_id, ...], ...}
        for seq_id in sequence_ids:
            try:
                # Only fetch minimal fields (id + sequence_set_id)
                seq_obj = Sequence.objects.only('id', 'sequence_set_id').get(id=seq_id)
            except Sequence.DoesNotExist:
                raise ValidationError(f"Sequence with ID {seq_id} does not exist.")

            set_groups[seq_obj.sequence_set_id].append(seq_id)

        # Prepare a list to hold final results in the correct order
        final_results = [None] * len(sequence_ids)

        # 3) For each distinct SequenceSet, build the slice expression and annotate the subset of Sequences
        for set_id, seq_ids_in_this_set in set_groups.items():
            # 3a) Retrieve the SequenceSet and its feature_dict
            try:
                seq_set = SequenceSet.objects.get(pk=set_id)
            except SequenceSet.DoesNotExist:
                raise ValidationError(f"SequenceSet with ID {set_id} does not exist.")

            feature_dict = seq_set.feature_dict

            # 3b) Convert the feature_list -> a list of column indices
            indices = []
            for feature in feature_list:
                if feature not in feature_dict:
                    raise ValidationError(
                        f"Feature '{feature}' not found in feature_dict of SequenceSet ID {set_id}"
                    )
                indices.append(feature_dict[feature])

            # 3c) Build the slice SQL expression
            if indices:
                # Example: ARRAY[sequence_data[1:1], sequence_data[2:2], ...]
                slice_sql = "ARRAY[" + ", ".join(
                    [f"sequence_data[{i + 1}:{i + 1}]" for i in indices]
                ) + "]"
            else:
                # If no features, produce a typed empty array to avoid PostgreSQL type errors
                slice_sql = "ARRAY[]::double precision[]"

            # 3d) Build the queryset for the subset of sequences matching these IDs & optional date filters
            subset_qs = Sequence.objects.filter(id__in=seq_ids_in_this_set)
            if start_date:
                subset_qs = subset_qs.filter(start_timestamp__gte=start_date)
            if end_date:
                subset_qs = subset_qs.filter(end_timestamp__lte=end_date)

            # 3e) Annotate with the slice expression
            annotated_qs = subset_qs.annotate(
                sliced_data=RawSQL(slice_sql, [])
            ).values(
                'id',
                'sequence_set',
                'start_timestamp',
                'end_timestamp',
                'sequence_length',
                'sliced_data'
            ).order_by('start_timestamp')

            # 3f) Process each record: convert None -> np.nan, shape the data via numpy
            subset_list = list(annotated_qs)
            for record in subset_list:
                # Convert None -> np.nan
                record['sliced_data'] = [
                    [np.nan if val is None else val for val in row]
                    for row in record['sliced_data']
                ]
                arr = np.array(record['sliced_data'])

                # Step 1: squeeze away trivial dims, e.g. (F,1,T)->(F,T) or (1,1,T)->(T,)
                arr = arr.squeeze()

                # Step 2: if it's 1D, that means we have shape (T,). Make it (1, T).
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, axis=0)

                # Now shape is (F, T) if multi-feature, or (1, T) if single-feature.

                # Step 3: transpose so we get (T, F) or (T, 1).
                arr = arr.T

                record['sliced_data'] = arr.tolist()
                record['metadata'] = seq_set.metadata

                # Place this record in final_results at the correct index
                original_pos = id_to_index[record['id']]
                final_results[original_pos] = record

        return final_results


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






