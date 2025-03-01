from abc import ABC

import numpy as np
import yfinance as yf
import pandas as pd
from dataset_manager.models import FeatureFactoryConfig, DataSet, DataRow
from django.db import transaction 
from dataset_manager.stock_config import STOCK_FACTORY_CONFIG_LIST
import importlib
import queue

MAX_LOOKBACK = 250

class FeatureFactoryService(ABC):

    def get_config_list(self):
        pass

    def get_example_df(self):
        pass


    def apply_feature_factories(self, df):
        factories = self.load_factories_from_db()
        for factory in factories:
            print(f"Applying {factory.config.name} to {df.index[0]} - {df.index[-1]}")
            df = factory.add_features(df)

        return df

    def load_factories_from_db(self):
        self.update_factory_configs()
        factories = []

        config_list = self.get_config_list()
        for config_name in config_list:
            config = FeatureFactoryConfig.objects.filter(name=config_name["name"]).first()
            module_name, class_name = config.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            factory_class = getattr(module, class_name)
            factory = factory_class(config)
            factories.append(factory)

        return factories

    def update_factory_configs(self):
        updated_configs_queue = queue.Queue()
        config_list = self.get_config_list()
        for config in config_list:
            db_config = FeatureFactoryConfig.objects.filter(name=config["name"]).first()

            if db_config:
                if db_config.version != config["version"]:
                    db_config.description = config["description"]
                    db_config.version = config["version"]
                    db_config.parameters = config["parameters"]
                    db_config.class_path = config["class_path"]
                    updated_configs_queue.put(db_config)
            else:
                db_config = FeatureFactoryConfig.objects.create(
                    name=config["name"],
                    description=config["description"],
                    version=config["version"],
                    parameters=config["parameters"],
                    class_path=config["class_path"]
                )
                updated_configs_queue.put(db_config)

        if updated_configs_queue.qsize() > 0:
            print("Updating feature factory configs")
            example_df = self.get_example_df()
        else:
            return

        updated_config_names = [config.name for config in list(updated_configs_queue.queue)]
        for config in config_list:
            db_config = FeatureFactoryConfig.objects.filter(name=config["name"]).first()
            module_name, class_name = db_config.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            factory_class = getattr(module, class_name)

            if config["name"] in updated_config_names:
                db_config_updated = updated_configs_queue.get()
                factory = factory_class(db_config_updated)
                example_df = factory.create_feature_names(example_df)
                print(f"Updated config for {db_config.name}")
                db_config_updated.save()
            else:
                factory = factory_class(db_config)
                example_df = factory.add_features(example_df)


class StockFeatureFactoryService(FeatureFactoryService):
    def get_config_list(self):
        return STOCK_FACTORY_CONFIG_LIST

    def get_example_df(self):
        return StockDataSetService.retreive_external_df(ticker="AAPL", start_date="2020-01-01", interval="1d")

    def get_all_x_features(self):
        all_configs = FeatureFactoryConfig.objects.all()
        x_features = []

        feature_names = [
            feature_name
            for config in all_configs
            if config.name != "TargetFeatureFactory"
            for feature_name in config.feature_names
        ]
        feature_names += ['open', 'high', 'low', 'close', 'volume']

        print(feature_names)
        id = 0
        for feature in feature_names:
            x_features.append({"id": id, "name": feature})
            id += 1

        return x_features


    def get_all_y_features(self):
        all_configs = FeatureFactoryConfig.objects.all()
        y_features = []

        id = 0
        for config in all_configs:
            if config.name == "TargetFeatureFactory":
                for feature in config.feature_names:
                    y_features.append({"id": id, "name": feature})
                    id += 1

        return y_features


class DataSetService(ABC):

    @staticmethod
    def retreive_external_df():
        pass

    @staticmethod
    def get_feature_factory_service():
        pass

    @classmethod
    def create_new_dataset(cls, start_date, end_date = None, **kwargs):
        if DataSet.objects.filter(dataset_type=kwargs.get('dataset_type'),
                                  metadata__contains = kwargs).exists():
            print(f"Dataset {kwargs.get('dataset_type')} already exists in the database")
            return

        try:
            df = cls.retreive_external_df(start_date, end_date, **kwargs)
        except Exception as e:
            print(f"Error fetching external data: {e}")
            return

        if len(df) < MAX_LOOKBACK:
            raise ValueError("Not enough data points to create a new dataset")

        df = cls.get_feature_factory_service().apply_feature_factories(df)

        with transaction.atomic():
            dataset = DataSet.objects.create(
                dataset_type=kwargs.get('dataset_type'),
                start_timestamp=df.index[0],
                end_timestamp=df.index[-1],
                features=df.columns.tolist(),
                metadata=kwargs,
            )

            dataset.save()
            DataSetService.dataframe_to_datarows(df, dataset)

        return df

    @staticmethod
    def get_data_set(dataset_type, **kwargs):
        return DataSet.objects.filter(dataset_type= dataset_type, metadata__contains=kwargs).first()

    @staticmethod
    def get_df_range(dataset, start_timestamp, end_timestamp):
        df = DataSetService.datarows_to_dataframe(dataset, start_timestamp, end_timestamp)
        return df

    @staticmethod
    def update_existing_dataset(df, dataset):
        '''
        Update an existing DataSet object with new data
        '''
        existing_records = DataRow.objects.filter(dataset=dataset)

        existing_records_map = {record.timestamp: record for record in existing_records}

        upddated_records = []

        df.fillna(-999, inplace=True)

        for index, row in df.iterrows():
            timestamp = index
            data_record = existing_records_map.get(timestamp)

            if data_record:
                data_record.features = row.to_dict()
                upddated_records.append(data_record)

            else:
                DataRow.objects.create(
                    dataset=dataset,
                    timestamp=timestamp,
                    features=row.to_dict(),
                )

        if upddated_records:
            with transaction.atomic():
                DataRow.objects.bulk_update(upddated_records, ['features'])

    @classmethod
    def update_recent_data(cls, dataset):
        '''
        Update the most recent data in the database
        '''
        last_data_point = DataRow.objects.filter(dataset=dataset).order_by('-timestamp').first()
        first_data_point = DataRow.objects.filter(dataset=dataset).order_by('timestamp').first()
        try:
            # add start_timestamp to meta_data with key 'start_date'

            new_data = cls.retreive_external_df(start_date = first_data_point.timestamp, **dataset.metadata)
        except Exception as e:
            print(f"Error fetching data for {dataset.dataset_type}: {e}")
            return

        # historical_df = DataSetService.datarows_to_dataframe(dataset)
        #
        # combined_df = pd.concat([historical_df[:-1][new_data.columns], new_data])
        combined_df = new_data

        df = cls.get_feature_factory_service().apply_feature_factories(combined_df)
        dataset.end_timestamp = df.index[-1]

        with transaction.atomic():
            DataSetService.update_existing_dataset(df, dataset)
            dataset.save()

    @staticmethod
    def add_new_features_to_all():
        '''
        Add a new feature to all existing datasets
        '''
        datasets = DataSet.objects.all()

        with transaction.atomic():
            for dataset in datasets:
                DataSetService.add_new_features(dataset)

    @classmethod
    def add_new_features(cls,dataset):
        '''
        Add a new feature to the existing dataset
        '''
        df = DataSetService.datarows_to_dataframe(dataset)
        df = cls.get_feature_factory_service().apply_feature_factories(df)

        DataSetService.update_existing_data(df, dataset)


    @staticmethod
    def dataframe_to_datarows(df, dataset):
        '''
        Save a DataFrame to the database as DataRow objects
        '''
        # replace nans with -999
        df.fillna(-999, inplace=True)

        with transaction.atomic():
            for _, row in df.iterrows():
                DataRow.objects.create(
                    dataset=dataset,
                    timestamp=row.name,
                    features=row.to_dict(),
                )

    @staticmethod
    def datarows_to_dataframe(dataset, start_timestamp=None, end_timestamp=None):
        '''
        Retrieve DataRow objects from the database and convert them to a DataFrame
        '''

        data_rows = DataRow.objects.filter(dataset=dataset)
        if start_timestamp:
            data_rows = data_rows.filter(timestamp__gte=start_timestamp)
        if end_timestamp:
            data_rows = data_rows.filter(timestamp__lte=end_timestamp)

        # Order the queryset by timestamp to ensure the correct order
        data_rows = data_rows.order_by('timestamp')

        # Convert to DataFrame
        df = pd.DataFrame([x.features for x in data_rows])
        df.index = [x.timestamp for x in data_rows]
        #set name to Date
        df.index.name = 'Date'

        df.replace({-999: np.nan}, inplace=True)

        return df


class StockDataSetService(DataSetService):

    @staticmethod
    def retreive_external_df(start_date, end_date = None, **kwargs):
        '''
        Retrieve stock data from the yfinance package
        '''
        ticker = kwargs.get('ticker')
        interval = kwargs.get('interval')

        df = pd.DataFrame()
        if end_date:
            df = yf.download([ticker], start=start_date, end=end_date, interval=interval, multi_level_index=False)
        else:
            df = yf.download([ticker], start=start_date, interval=interval, multi_level_index=False)

        print(df.head())
        if df.index.tzinfo is None:
            df.index = pd.to_datetime(df.index).tz_localize('UTC')

        df.index = pd.to_datetime(df.index).tz_convert('UTC')

        print(df.head())

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return df

    @staticmethod
    def get_feature_factory_service():
        return StockFeatureFactoryService()












