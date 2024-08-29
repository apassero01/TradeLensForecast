from sequenceset_manager.models import StockSequence, FeatureDict
from django.db import transaction
import requests
import pandas as pd
from django.db.models.expressions import RawSQL
from django.db.models import Case, When, Value, F, FloatField, Func
from django.db import models
from django.core.exceptions import ValidationError
from django.db import connection
import numpy as np

class SequencesetManagerService:

    # def update_recent_sequences(ticker, interval, sequence_length):
    #     '''
    #     Update the most recent sequences for a given stock ticker
    #     '''
    #     # get most recent sequence
    #     most_recent_seq = StockSequence.objects.filter(ticker=ticker, timeframe=interval).order_by('-end_timestamp').first()
    #     if most_recent_seq is None:
    #         print(f"No sequences found for {ticker}")
    #         return None
        
    #     # get most recent date
    #     most_recent_date = most_recent_seq.start_timestamp


    def create_new_sequence_set(ticker, interval, sequence_length, start_date = None, end_date = None):
        '''
        Create a new sequence set for a given stock ticker
        '''
        ticker = ticker.upper()
        # Check if the sequence set already exists
        existing_sequences = StockSequence.objects.filter(ticker=ticker, timeframe=interval)
        if existing_sequences.exists():
            print(f"Sequence set for {ticker} already exists")
            return
        df = SequencesetManagerService.get_stock_dataset(ticker, interval, start_date, end_date)
        if df is None:
            print(f"Error fetching stock data for {ticker}")
            return None
        print("Creating Sequence objects")
        stock_sequences, feature_dict = SequencesetManagerService.create_sequence_objects(ticker, interval, sequence_length, df)
        print("Saving Sequence objects")
        SequencesetManagerService.save_stock_sequences(stock_sequences, feature_dict)
        return stock_sequences

    def create_sequence_objects(ticker, timeframe, sequence_length, df):
        '''
        Create a 3D sequence from a DataFrame
        '''
        ticker = ticker.upper()
        
        df_cols = df.columns
        df_cols = sorted(df_cols)
        df = df[df_cols]

        timestamps = df.index
        df_values = df.values

        indices_seq = list(range(len(df_cols)))

        feature_dict = {col: index for col, index in zip(df_cols, indices_seq)}

        stock_sequences = []

        for i in range(len(df_values) - 1, -1, -1):
            start_index = i - sequence_length + 1
            if start_index < 0:
                break
            
            end_idx = i

            seq = df_values[start_index:end_idx + 1, :]
            start_date = timestamps[start_index]
            end_date = timestamps[end_idx]

            stock_sequences.append(StockSequence(
                ticker=ticker,
                start_timestamp=start_date,
                end_timestamp=end_date,
                sequence_length=sequence_length,
                sequence_data=seq.tolist(),
                timeframe=timeframe,
            ))

        feature_dict_obj = FeatureDict(
            ticker=ticker,
            feature_dict=feature_dict,
            timeframe=timeframe
        )

        return stock_sequences, feature_dict_obj
    
    def save_stock_sequences(stock_sequences, feature_dict):
        '''
        Save a list of StockSequence objects to the database
        '''
        print("Transposing Sequences")
        for seq in stock_sequences:
            seq.sequence_data = SequencesetManagerService.transpose(seq.sequence_data)
        print("Saving Sequences")
        batch_size = 1000
        with transaction.atomic():
            for i in range(0, len(stock_sequences), batch_size):
                StockSequence.objects.bulk_create(stock_sequences[i:i+batch_size])
            feature_dict.save()
    
    def get_stock_dataset(ticker, interval = "1d", start_date = None, end_date = None):
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
    

    @staticmethod
    def retrieve_sequence_slice(ticker, features, interval= '1d', start_date=None, end_date=None, sequence_length=None):
        queryset = StockSequence.objects.filter(ticker = ticker)
        print(queryset.count())

        ## TODO Fix what I am about to write 
        column_dict = FeatureDict.objects.get(ticker=ticker).feature_dict
        indices = [column_dict[feature] for feature in features]

        print(interval)
        queryset = queryset.filter(timeframe=interval)
        print(queryset.count())

        if start_date:
            queryset = queryset.filter(start_timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(end_timestamp__lte=end_date)
        if sequence_length:
            queryset = queryset.filter(sequence_length=sequence_length)

        print(queryset.count())
        # Prepare the SQL for selecting specific indices
        queryset = queryset.annotate(
            sliced_data=RawSQL(
                f"ARRAY[{', '.join([f'sequence_data[{i+1}:{i+1}]' for i in indices])}]",
                []
            )
        ).values('id', 'ticker', 'start_timestamp', 'end_timestamp', 'sequence_length', 'sliced_data')

        print(queryset.count())
        result = list(queryset.order_by('start_timestamp'))

        # Flatten and transpose the array
        for record in result:
            record['sliced_data'] = np.squeeze(np.array(record['sliced_data']))
            # print(record['sliced_data'].shape)
            record['sliced_data'] = record['sliced_data'].T.tolist()
            # print(record['sliced_data'].shape)

        return result


    
    def transpose(matrix_list):
        '''
        Transpose a list of matrices
        '''
        return [list(row) for row in zip(*matrix_list)]

        


