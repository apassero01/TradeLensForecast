import yfinance as yf
import pandas as pd
from datetime import datetime
from dataset_manager.models import StockData, FeatureFactoryConfig
from dataset_manager.factories import MovingAverageFeatureFactory
from django.db import transaction 
from dataset_manager.config import FACTORY_CONFIG_LIST
import importlib
import json
import queue
from django.core.cache import cache

MAX_LOOKBACK = 250

class DatasetManagerService:

    @staticmethod
    def create_new_stock(ticker: str, start_date = None, end_date=None, interval="1d"):
        '''
        Create a new stock in the database
        '''
        if StockData.objects.filter(ticker=ticker).filter(timeframe=interval).exists():
            print(f"Stock {ticker} and {interval} already exists in the database")
            return
        
        ticker = ticker.upper()
        try:
            df, _ = DatasetManagerService.fetch_stock_data(ticker, start_date, end_date, interval)
            print(df.head())
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return

        if len(df) < MAX_LOOKBACK:
            print("len(df):", len(df))
            raise ValueError("Not enough data points to create a new stock")
        df = FeatureFactoryService.apply_feature_factories(df)
        print(df.columns)
        DatasetManagerService.dataframe_to_stockdata(df, ticker, interval)

        return df

    @staticmethod
    def update_recent_stock_data(ticker: str, interval="1d"):
        '''
        Update the most recent stock data in the database
        '''

        last_data_point = StockData.objects.filter(ticker=ticker).order_by('-timestamp').first()

        try:
            new_data, _ = DatasetManagerService.fetch_stock_data(ticker, last_data_point.timestamp , None, interval)
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return

        if last_data_point: 
            historical_df = DatasetManagerService.stockdata_to_dataframe(ticker, interval)
            historical_df = historical_df.tail(MAX_LOOKBACK)

            combined_df = pd.concat([historical_df[:-1], new_data])
        
        else:
            combined_df = new_data
        
        df = FeatureFactoryService.apply_feature_factories(combined_df)
        DatasetManagerService.dataframe_to_stockdata(df.tail(len(new_data)-1), ticker, interval)

    @staticmethod
    def add_new_feature(ticker: str, interval="1d"):
        '''
        Add a new feature to the existing stock data
        '''
        ticker = ticker.upper()
        df = DatasetManagerService.stockdata_to_dataframe(ticker, interval)
        df = FeatureFactoryService.apply_feature_factories(df)
        
        DatasetManagerService.update_existing_stock_data(df, ticker, interval)


    @staticmethod
    def update_existing_stock_data(df : pd.DataFrame, ticker: str, interval="1d"):
        '''
        Update an existing StockData object with new data
        '''
        existing_records = StockData.objects.filter(ticker=ticker, timeframe=interval)

        existing_records_map = {record.timestamp: record for record in existing_records}

        upddated_records = [] 

        for index, row in df.iterrows():
            timestamp = row.name
            stock_record = existing_records_map.get(timestamp)

            if stock_record:
                stock_record.features = row.to_dict()
                upddated_records.append(stock_record)
            
        if upddated_records:
            with transaction.atomic():
                StockData.objects.bulk_update(upddated_records, ['features'])

    @staticmethod
    def dataframe_to_stockdata(df, ticker, timeframe):
        '''
        Save a DataFrame to the database as StockData objects
        '''

        with transaction.atomic():
            for _, row in df.iterrows():
                # try:
                #     json_str = json.dumps(row)
                # except TypeError as e:
                #     print("Invalid JSON:", e)
                    # print(row.to_dict())

                StockData.objects.create(
                    ticker=ticker.upper(),
                    timestamp=row.name,
                    timeframe=timeframe,
                    features=row.to_dict(),
                )
        
    @staticmethod
    def stockdata_to_dataframe(ticker=None, timeframe=None, start_date=None, end_date=None) -> pd.DataFrame:
        '''
        Retrieve StockData objects from the database and convert them to a DataFrame
        '''

        stock_data = StockData.objects.all()
        if ticker:
            stock_data = stock_data.filter(ticker=ticker)
        if timeframe:
            stock_data = stock_data.filter(timeframe=timeframe)
        if start_date:
            stock_data = stock_data.filter(timestamp__gte=start_date)
        if end_date:
            stock_data = stock_data.filter(timestamp__lte=end_date)
        
        # Order the queryset by timestamp to ensure the correct order
        stock_data = stock_data.order_by('timestamp')
        
        # Convert to DataFrame
        df = pd.DataFrame([x.features for x in stock_data])
        df.index = [x.timestamp for x in stock_data]

        df.replace({-999: None}, inplace=True)
        
        return df
    
    @staticmethod
    def fetch_stock_data(
        ticker: str, start_date: datetime, end_date: datetime, interval="1d"
    ) -> pd.DataFrame:
        """
        Use the yfinance package and read the requested ticker from start_date to end_date. The following additional
        variables are created:

        additional variables: binarized weekday, month, binarized q1-q4

        All scaled variables are

        :param ticker:
        :param start_date:
        :param end_date:
        :param interval: 1d, 1wk, 1mo etc consistent with yfinance api
        :return:
        """

        df = pd.DataFrame()
        if end_date:
            df = yf.download([ticker], start=start_date, end=end_date, interval=interval)
        else:
            df = yf.download([ticker], start=start_date, interval=interval)

        df = df.drop(columns="Adj Close")

        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d 00:00:00+00:00')
        df.index = pd.to_datetime(df.index)  # Convert back to datetime with the correct format
        # Standard column names needed for pandas-ta
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        return df, df.columns.tolist()


class FeatureFactoryService:

    def tearDown(self):
        # Clean up tasks after each test
        StockData.objects.all().delete()  # Clear the model's data if necessary
        FeatureFactoryConfig.objects.all().delete()  # Clear the model's data if necessary
        cache.clear()  # Clear the cache

    @staticmethod
    def apply_feature_factories(df):
        factories = FeatureFactoryService.load_factories_from_db()
        for factory in factories:
            print(f"Applying {factory.config.name} to {df.index[0]} - {df.index[-1]}")
            df = factory.add_features(df)
        
        return df

    @staticmethod
    def load_factories_from_db():

        FeatureFactoryService.update_factory_configs()
        factories = []
        configs = FeatureFactoryConfig.objects.all()

        for config in configs:
            module_name, class_name = config.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            factory_class = getattr(module, class_name)
            factory = factory_class(config)
            factories.append(factory)
            
        return factories

    @staticmethod
    def update_factory_configs():
        updated_configs_queue = queue.Queue()
        for config in FACTORY_CONFIG_LIST:
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
            example_df, _ = DatasetManagerService.fetch_stock_data("AAPL", "2021-01-01", "2021-01-10")
        else:
            return

        updated_config_names = [config.name for config in list(updated_configs_queue.queue)]
        for config in FACTORY_CONFIG_LIST:
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
            








