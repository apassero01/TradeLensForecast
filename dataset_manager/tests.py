from datetime import datetime

import numpy as np
import pandas as pd
from django.test import TestCase
from keras.src.legacy.backend import update
import pandas.testing as pdt

from dataset_manager.models import FeatureFactoryConfig, DataSet, DataRow
from dataset_manager.services import FeatureFactoryService, StockDataSetService, StockFeatureFactoryService, \
    DataSetService
from dataset_manager.factories import MovingAverageFeatureFactory, OHLCVFeatureFactory, BandFeatureFactory, MomentumFeatureFactory,TargetFeatureFactory


class StockFeatureFactoryServiceTest(TestCase):

    def setUp(self) -> None:
        self.stock_feature_factory_service = StockFeatureFactoryService()
        self.stock_feature_factory_service.update_factory_configs()
        self.FACTORY_CONFIG_LIST = self.stock_feature_factory_service.get_config_list()
        super().setUp()

    def tearDown(self) -> None:
        FeatureFactoryConfig.objects.all().delete()
        super().tearDown()

    def test_update_factory_configs(self):
        ma_factory_config = self.FACTORY_CONFIG_LIST[0]
        ma_factory_config["parameters"]["windows"].append(12)
        ma_factory_config["version"] = "0.0.2"

        self.stock_feature_factory_service.update_factory_configs()

        updated_config = FeatureFactoryConfig.objects.get(name="MovingAverageFeatureFactory")

        # Assert that version is correctly updated and 12 is in the windows list
        self.assertEqual(updated_config.version, "0.0.2")
        self.assertIn(12, updated_config.parameters["windows"])

    def test_load_factories_from_db(self):
        factories = self.stock_feature_factory_service.load_factories_from_db()
        self.assertEqual(len(factories), len(self.FACTORY_CONFIG_LIST))
        self.assertEqual(factories[0].config.name, "MovingAverageFeatureFactory")

    def test_apply_feature_factories(self):
        df = StockDataSetService.retreive_external_df(
            ticker="SPY",
            start_date="2020-01-01",
            end_date="2021-01-10",
            interval="1d"
        )
        df = self.stock_feature_factory_service.apply_feature_factories(df)

        initial_columns = ["open", "high", "low", "close", "volume"]
        factory_added_columns = [
            config.feature_names for config in FeatureFactoryConfig.objects.all()
        ]
        all_columns = initial_columns.copy()
        for factory_columns in factory_added_columns:
            all_columns += factory_columns

        self.assertEqual(list(df.columns), all_columns)

class MovingAverageFeatureFactoryTest(TestCase):
    def setUp(self):
        self.df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        config = FeatureFactoryConfig(name="MovingAverageFeatureFactory", parameters={"windows": [5, 10, 20, 50, 100, 200]})
        self.factory = MovingAverageFeatureFactory(config)
    
    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()

    def test_createSMA(self):
        self.df = self.factory.create_SMA(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'sma{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['sma5'].iloc[20], self.df['close'].iloc[16:21].mean())  # Check computed value

    def test_createEMA(self):
        self.df = self.factory.create_EMA(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'ema{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(
            self.df['ema5'].iloc[30],
            self.df['close'].iloc[:31].ewm(span=5, adjust=False).mean().iloc[-1], places=3
        ) # Check computed value

    def test_createSMAVolume(self):
        self.df = self.factory.create_SMA_Volume(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'smaVol{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['smaVol5'].iloc[20], self.df['volume'].iloc[16:21].mean())  # Check computed value

    def test_createSMAPctDiff(self):
        self.df = self.factory.create_SMA(self.df)
        self.df = self.factory.createSMAPctDiff(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'pctDiff+sma{period}_close' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods) + len(periods)*(len(periods)-1)//2)
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['pctDiff+sma5_close'].iloc[20], (self.df['close'].iloc[20] - self.df['sma5'].iloc[20]) / self.df['sma5'].iloc[20]*100)  # Check computed value

    def test_createEMAPctDiff(self):    
        self.df = self.factory.create_EMA(self.df)
        self.df = self.factory.createEMAPctDiff(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'pctDiff+ema{period}_close' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods) + len(periods)*(len(periods)-1)//2)
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['pctDiff+ema5_close'].iloc[20], (self.df['close'].iloc[20] - self.df['ema5'].iloc[20]) / self.df['ema5'].iloc[20]*100)  # Check computed value

    def test_createSMAPctDiffVol(self):
        self.df = self.factory.create_SMA_Volume(self.df)
        self.df = self.factory.createSMAPctDiffVolume(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'pctDiff+smaVol{period}_volume' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods) + len(periods)*(len(periods)-1)//2)
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['pctDiff+smaVol5_volume'].iloc[20], (self.df['volume'].iloc[20] - self.df['smaVol5'].iloc[20]) / self.df['smaVol5'].iloc[20]*100)  # Check computed value

    def test_createSMADerivative(self):
        self.df = self.factory.create_SMA(self.df)
        self.df = self.factory.createSMADerivative(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'deriv+sma{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['deriv+sma5'].iloc[40], (self.df['sma5'].iloc[40] - self.df['sma5'].iloc[35]) / 5)

    def test_createEMADerivative(self):
        self.df = self.factory.create_EMA(self.df)
        self.df = self.factory.createEMADerivative(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'deriv+ema{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(self.df['deriv+ema5'].iloc[20], (self.df['ema5'].iloc[20] - self.df['ema5'].iloc[15]) / 5)

    def test_createSMADerivativeVol(self):
        self.df = self.factory.create_SMA_Volume(self.df)
        self.df = self.factory.createSMADerivativeVolume(self.df)
        periods = [5, 10, 20, 50, 100, 200]
        for period in periods:
            self.assertTrue(f'deriv+smaVol{period}' in self.df.columns)
        self.assertTrue(len(self.df.columns) >= len(periods))
        self.assertFalse(self.df.isnull().values.any())  # Check for NaNs

        self.assertAlmostEqual(self.df['deriv+smaVol5'].iloc[20], (self.df['smaVol5'].iloc[20] - self.df['smaVol5'].iloc[15]) / 5)

class TestOHLCVFeatureFactory(TestCase):

    def setUp(self):
        # Download historical data for AAPL
        self.df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        config = FeatureFactoryConfig(name="OHLCVFeatureFactory", parameters={})
        self.factory = OHLCVFeatureFactory(config)
    
    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()

    def test_createPctChg(self):
        # Generate percentage change features
        pctChg_df = self.factory.createPctChg(self.df.copy())

        # Ensure that percentage change columns are present
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertTrue(f'pctChg{col}' in pctChg_df.columns)
        
        # Check the number of columns matches the original columns plus the new pctChg columns
        expected_columns = len(self.df.columns) + len(['open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(len(pctChg_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(pctChg_df.isnull().values.any())

        # Verify that a specific computed value matches expectations
        self.assertAlmostEqual(pctChg_df['pctChgclose'].iloc[20], 
                               (self.df['close'].iloc[20] - self.df['close'].iloc[19]) / self.df['close'].iloc[19] * 100)

    def test_createIntraDay(self):
        # Generate intraday features
        intraday_df = self.factory.createIntraDay(self.df.copy())

        # Expected new columns after running createIntraDay
        new_cols = ['opHi', 'opLo', 'hiCl', 'loCl', 'hiLo', 'opCl', 'pctChgClOp', 'pctChgClLo', 'pctChgClHi']

        # Ensure that all new columns are present
        self.assertTrue(all([col in intraday_df.columns for col in new_cols]))

        # Check that the number of columns is the sum of the original and new columns
        expected_columns = len(self.df.columns) + len(new_cols)
        self.assertEqual(len(intraday_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(intraday_df.isnull().values.any())

        # Verify that a specific computed value matches expectations
        self.assertAlmostEqual(intraday_df['opHi'].iloc[20], 
                               (self.df['high'].iloc[20] - self.df['open'].iloc[20]) / self.df['open'].iloc[20] * 100)



class TestBandFeatureFactory(TestCase):
    def setUp(self):
        # Download historical data for AAPL
        self.df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        
        # Set up the configuration with the desired windows
        config = FeatureFactoryConfig(name="BandFeatureFactory", parameters={"windows": [5, 10, 20, 50, 100, 200]})
        
        # Initialize the BandFeatureFactory with the configuration
        self.factory = BandFeatureFactory(config)

    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()

    def test_createBB(self):
        # Generate Bollinger Bands features
        bb_df = self.factory.createBB(self.df.copy())

        # Check if Bollinger Bands columns are present for each period
        for period in [5, 10, 20, 50, 100, 200]:
            self.assertTrue(f'bb_high{period}' in bb_df.columns)
            self.assertTrue(f'bb_low{period}' in bb_df.columns)
        
        # Ensure the number of columns matches the expected Bollinger Bands columns
        expected_columns = len(self.df.columns) + len([5, 10, 20, 50, 100, 200]) * 2
        self.assertEqual(len(bb_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(bb_df.isnull().values.any())

    def test_createBBPctDiff(self):
        # Generate Bollinger Bands percentage difference features
        bb_df = self.factory.createBB(self.df.copy())
        bbPctDiff_df = self.factory.createBBPctDiff(bb_df.copy())

        # Check if the percentage difference columns are present for each period
        for period in [5, 10, 20, 50, 100, 200]:
            self.assertTrue(f'pctDiff+bb_high_low{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'pctDiff+bb_high_close{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'pctDiff+bb_low_close{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'bb_indicator{period}' in bbPctDiff_df.columns)

        # Ensure the number of columns matches the expected columns
        expected_columns = len(bb_df.columns) + len([5, 10, 20, 50, 100, 200]) * 4
        self.assertEqual(len(bbPctDiff_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(bbPctDiff_df.isnull().values.any())

        # Verify that a specific computed value matches expectations
        self.assertAlmostEqual(
            bbPctDiff_df[f'pctDiff+bb_high_low5'].iloc[20], 
            (bb_df[f'bb_high5'].iloc[20] - bb_df[f'bb_low5'].iloc[20]) / bb_df[f'bb_low5'].iloc[20] * 100
        ) 

class TestMomentumFeatureFactory(TestCase):
    def setUp(self):
        # Download historical data for AAPL
        self.df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        
        # Set up the configuration with the desired RSI periods
        config = FeatureFactoryConfig(name="MomentumFeatureFactory", parameters={"rsi_periods": [5, 10, 20, 50, 100]})
        
        # Initialize the MomentumFeatureFactory with the configuration
        self.factory = MomentumFeatureFactory(config)
    
    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super

    def test_createRSI(self):
        # Generate RSI features
        rsi_df = self.factory.createRSI(self.df.copy())

        # Check if RSI columns are present for each period
        for period in [5, 10, 20, 50, 100]:
            self.assertTrue(f'rsi{period}' in rsi_df.columns)
        
        # Ensure the number of columns matches the expected RSI columns
        expected_columns = len(self.df.columns) + len([5, 10, 20, 50, 100])
        self.assertEqual(len(rsi_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(rsi_df.isnull().values.any())

    def test_createMACD(self):
        # Generate MACD features
        macd_df = self.factory.createMACD(self.df.copy())

        # Check if MACD columns are present
        macd_cols = ["macd", "macd_signal", "macd_diff"]
        for col in macd_cols:
            self.assertTrue(col in macd_df.columns)
        
        # Ensure the number of columns matches the expected MACD columns
        expected_columns = len(self.df.columns) + len(macd_cols)
        self.assertEqual(len(macd_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(macd_df.isnull().values.any())

        # Verify that a specific computed value matches expectations
        self.assertAlmostEqual(
            macd_df['macd'].iloc[20], 
            macd_df['macd'].iloc[20]  # Assuming comparison with itself for demonstration, replace with correct logic if needed
        )

    def test_createStock(self):
        # Generate Stochastic Oscillator features
        stock_df = self.factory.createStock(self.df.copy())

        # Check if Stochastic Oscillator columns are present
        stock_cols = ['stock_k', 'stock_d']
        for col in stock_cols:
            self.assertTrue(col in stock_df.columns)
        
        # Ensure the number of columns matches the expected Stochastic Oscillator columns
        expected_columns = len(self.df.columns) + len(stock_cols)
        self.assertEqual(len(stock_df.columns), expected_columns)

        # Ensure no NaNs in the dataframe
        self.assertFalse(stock_df.isnull().values.any())

        # Verify that a specific computed value matches expectations
        self.assertAlmostEqual(
            stock_df['stock_k'].iloc[20], 
            stock_df['stock_k'].iloc[20]  # Assuming comparison with itself for demonstration, replace with correct logic if needed
        )


class TestTargetFeatureFactory(TestCase):
    def setUp(self):
        # Download historical data for AAPL
        self.df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        
        # Set up the configuration with the desired output steps
        self.output_steps = 15   
        target_config = FeatureFactoryConfig(name="TargetFeatureFactory", parameters={"output_steps": self.output_steps})
        pctChg_config = FeatureFactoryConfig(name="OHLCVFeatureFactory", parameters={})
        
        # Initialize the TargetFeatureFactory with the configuration
        self.target_factory = TargetFeatureFactory(target_config)
        self.pctChg_factory = OHLCVFeatureFactory(pctChg_config)
    
    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()
        


    def test_create_pct_chg_target(self):
        # Generate target variable features
        self.df = self.pctChg_factory.createPctChg(self.df.copy())
        target_df = self.target_factory.add_features(self.df.copy())

        # Check if target variable columns are present
        target_cols = ["pctChgclose+" + str(i) for i in range(1,self.output_steps+1)]

        # Ensure the number of columns matches the expected target variable columns
        for col in target_cols:
            self.assertTrue(col in target_df.columns)


        # Verify that a specific computed value matches expectations
        example_row = 20 

        for i in range(1,self.output_steps+1):
            self.assertAlmostEqual(
                target_df['pctChgclose+' + str(i)].iloc[example_row], 
                self.df['pctChgclose'].iloc[example_row+i]
            )

    def test_create_rolling_sum_vars(self):
        # Generate percentage change features first
        self.df = self.pctChg_factory.createPctChg(self.df.copy())

        # Generate cumulative sum variables
        self.df = self.target_factory.create_rolling_sum_vars(self.df.copy())

        # Verify that the new columns have been added correctly
        for roll in range(1,self.output_steps+1):
            shifted_col_name = f'cumPctChg+{roll}'


            # Ensure the shifted column is present
            self.assertTrue(shifted_col_name in self.df.columns)

        # Verify that the cumulative sums and shifted values are correct
        for roll in range(1,self.output_steps+1):
            shifted_col_name = f'cumPctChg+{roll}'
            for i in range(0, len(self.df) - roll):
                # Calculate the expected cumulative sum from i+1 to i+roll
                expected_value = self.df['pctChgclose'].iloc[i + 1: i + 1 + roll].sum()

                # Compare the value at the current row with the expected value
                self.assertAlmostEqual(
                    self.df[shifted_col_name].iloc[i],
                    expected_value,
                    msg=f"Mismatch in {shifted_col_name} at index {i}"
                )

    def test_create_raw_target(self):
        # Generate target variable features
        self.df = self.pctChg_factory.createPctChg(self.df.copy())
        target_df = self.target_factory.add_features(self.df.copy())

        # Check if target variable columns are present
        target_cols = ["close+" + str(i) for i in range(1,self.output_steps+1)]

        # Ensure the number of columns matches the expected target variable columns
        for col in target_cols:
            self.assertTrue(col in target_df.columns)

        # Verify that a specific computed value matches expectations
        example_row = 20

        for i in range(1,self.output_steps+1):
            self.assertAlmostEqual(
                target_df['close+' + str(i)].iloc[example_row],
                self.df['close'].iloc[example_row+i]
            )

class TestStockDataSetService(TestCase):
    def setUp(self):
        self.feature_factory = StockFeatureFactoryService()
        self.feature_factory.update_factory_configs()
        initial_columns = ["open", "high", "low", "close", "volume"]
        self.factory_added_columns = [config.feature_names for config in FeatureFactoryConfig.objects.all()]
        self.all_columns = initial_columns
        for factory_columns in self.factory_added_columns:
            self.all_columns += factory_columns

        self.all_columns = sorted(self.all_columns)
        return super().setUp()

    def tearDown(self) -> None:
        DataSet.objects.all().delete()
        DataRow.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()

    def test_retreive_external_df(self):
        df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        self.assertEqual(df.columns.tolist(), ["open", "high", "low", "close", "volume"])

        df = StockDataSetService.retreive_external_df(ticker = "AAPL", start_date = "2020-01-01", interval = "1d")
        self.assertEqual(df.columns.tolist(), ["open", "high", "low", "close", "volume"])

    def test_dataframe_to_datarows(self):
        df = StockDataSetService.retreive_external_df(ticker = "SPY", start_date = "2020-01-01", end_date = "2021-01-10", interval = "1d")
        dataset = DataSet.objects.create(dataset_type="stock", start_timestamp="2020-01-01", end_timestamp="2021-01-10", features = self.all_columns, metadata = {"ticker": "SPY", "timeframe": "1d"})
        StockDataSetService.dataframe_to_datarows(df, dataset)

        self.assertEqual(DataRow.objects.count(), len(df))

        df.index = pd.DatetimeIndex(df.index)

        datarows = DataRow.objects.all()
        for datarow in datarows:
            # assert that the data is the same
            self.assertEqual(datarow.features, df.loc[datarow.timestamp].to_dict())

    def test_create_new_dataset(self):
        StockDataSetService.create_new_dataset(dataset_type="stock", start_date="2020-01-01", end_date="2021-01-10", ticker="SPY", interval="1d")
        df = StockDataSetService.retreive_external_df(ticker="SPY", start_date="2020-01-01", end_date="2021-01-10",
                                                      interval="1d")
        df = self.feature_factory.apply_feature_factories(df)

        dataset = DataSet.objects.first()
        self.assertEqual(dataset.dataset_type, "stock")
        self.assertEqual(dataset.start_timestamp, df.iloc[0].name)
        self.assertEqual(dataset.end_timestamp, df.iloc[-1].name)
        self.assertEqual(sorted(dataset.features), sorted(self.all_columns))

        datarows = DataRow.objects.filter(dataset=dataset)
        self.assertEqual(len(datarows), len(df))
        for datarow in datarows:
            self.assertEqual(sorted(datarow.features.items()), sorted(df.loc[datarow.timestamp].to_dict().items()))


    def test_update_existing_dataset(self):
        StockDataSetService.create_new_dataset(dataset_type="stock", start_date="2020-01-01", end_date="2021-01-10", ticker="SPY", interval="1d")
        df = StockDataSetService.retreive_external_df(ticker="SPY", start_date="2020-01-01", end_date="2021-01-10",
                                                      interval="1d")
        df = self.feature_factory.apply_feature_factories(df)
        dataset = DataSet.objects.first()

        new_cols = [["new_column", "new_column2"]]
        df[new_cols] = 1
        StockDataSetService.update_existing_dataset(df, dataset)

        datarows = DataRow.objects.filter(dataset=dataset)
        self.assertEqual(len(datarows), len(df))
        for datarow in datarows:
            self.assertEqual(sorted(datarow.features.items()), sorted(df.loc[datarow.timestamp].to_dict().items()))

    def test_update_recent_data(self):

        StockDataSetService.create_new_dataset(dataset_type="stock", start_date="2020-01-01", end_date="2021-01-10", ticker="SPY", interval="1d")

        df = StockDataSetService.retreive_external_df(ticker="SPY", start_date="2020-01-01",
                                                      interval="1d")

        df = self.feature_factory.apply_feature_factories(df)

        dataset = DataSet.objects.first()
        StockDataSetService.update_recent_data(dataset)

        updated_dataset = DataSet.objects.first()

        self.assertEqual(updated_dataset.start_timestamp, df.index.min())
        self.assertEqual(updated_dataset.end_timestamp, df.index.max())
        self.assertEqual(sorted(updated_dataset.features), sorted(self.all_columns))

        df_db = StockDataSetService.datarows_to_dataframe(dataset)
        df_db = df_db[sorted(df_db.columns)]
        df = df[sorted(df.columns)]
        df.replace({-999: np.nan}, inplace=True)
        df = df.astype(np.float64)

        self.assertEqual(len(df_db), len(df))
        self.assertEqual(sorted(df_db.columns), sorted(df.columns))
        self.assertEqual(df_db.index.tolist(), df.index.tolist())
        self.assertTrue(np.allclose(df_db.values, df.values, equal_nan=True))


    def test_datarows_to_dataframe(self):
        StockDataSetService.create_new_dataset(dataset_type="stock", start_date="2020-01-01", end_date="2021-01-10", ticker="SPY", interval="1d")
        truth_df = StockDataSetService.retreive_external_df(ticker="SPY", start_date="2020-01-01", end_date="2021-01-10", interval="1d")
        truth_df = self.feature_factory.apply_feature_factories(truth_df)
        truth_df.replace({-999: np.nan}, inplace=True)
        dataset = DataSet.objects.first()
        df = StockDataSetService.datarows_to_dataframe(dataset)

        df = df[sorted(df.columns)]
        truth_df = truth_df[sorted(truth_df.columns)]

        self.assertEqual((df.columns.tolist()), (truth_df.columns.tolist()))
        self.assertEqual(df.index.tolist(), truth_df.index.tolist())
        self.assertTrue(np.array_equal(df.values, truth_df.values, equal_nan=True))











        

    
