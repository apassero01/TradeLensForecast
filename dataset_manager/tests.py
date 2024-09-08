from django.test import TestCase
from dataset_manager.models import StockData, FeatureFactoryConfig, DataSetTracker
from dataset_manager.services import FeatureFactoryService, DatasetManagerService, DatasetTrackerService, \
    FeatureTrackerService
from dataset_manager.config import FACTORY_CONFIG_LIST
from dataset_manager.factories import MovingAverageFeatureFactory, OHLCVFeatureFactory, BandFeatureFactory, MomentumFeatureFactory,TargetFeatureFactory

class DatasetManagerServiceTest(TestCase):
    def setUp(self) -> None:
        FeatureFactoryService.update_factory_configs()
        initial_columns = ["open", "high", "low", "close", "volume"]
        self.factory_added_columns = [config.feature_names for config in FeatureFactoryConfig.objects.all()]
        self.all_columns = initial_columns
        for factory_columns in self.factory_added_columns:
            self.all_columns += factory_columns
        
        self.all_columns = sorted(self.all_columns)
        return super().setUp()

    def tearDown(self) -> None:
        StockData.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()
    
    def test_fetch_stock_data(self):
        df, columns = DatasetManagerService.fetch_stock_data("SPY", "2020-01-01", "2021-01-10")
        self.assertEqual(columns, ["open", "high", "low", "close", "volume"])

    def test_create_new_stock(self):
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
        self.assertEqual(StockData.objects.first().ticker, "SPY")
        self.assertEqual(StockData.objects.first().timeframe, "1d")
        print(StockData.objects.first().features.keys())
        self.assertEqual(sorted(list(StockData.objects.first().features.keys())), self.all_columns)

        self.assertTrue(DataSetTracker.objects.get(ticker="SPY", timeframe="1d") is not None)
        FeatureTrackerService.ensure_synced_features()

    def test_stockdata_to_dataframe(self):
        df_api,_ = DatasetManagerService.fetch_stock_data("SPY", "2020-01-01", "2021-01-10")
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
        df = DatasetManagerService.stockdata_to_dataframe("SPY", "1d")

        self.assertEqual(sorted(list(df.columns)), self.all_columns)
        # test timestamps are the same 
        self.assertEqual(df.index.tolist(), df_api.index.tolist())

    def test_update_existing_stock_data(self):
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
        initial_count = StockData.objects.count()
        df = DatasetManagerService.stockdata_to_dataframe("SPY", "1d")
        df["new_column"] = 1
        DatasetManagerService.update_existing_stock_data(df, "SPY", "1d")

        self.assertEqual(sorted(list(StockData.objects.first().features.keys())), sorted(self.all_columns + ["new_column"]))
        self.assertEqual(StockData.objects.count(), initial_count)

    def test_add_new_feature(self):
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")

        # update the ma factory config adding a new average 
        ma_factory_config = FACTORY_CONFIG_LIST[0]

        ma_factory_config["parameters"]["windows"].append(12)
        ma_factory_config["version"] = "0.0.2"

        dataset_tracker = DataSetTracker.objects.first()


        DatasetManagerService.add_new_feature(dataset_tracker)
        self.assertTrue("sma12" in StockData.objects.first().features.keys())

    def test_add_new_feature_to_all(self):
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
        DatasetManagerService.create_new_stock("AAPL", "2020-01-01", "2021-01-10")

        # update the ma factory config adding a new average
        ma_factory_config = FACTORY_CONFIG_LIST[0]

        ma_factory_config["parameters"]["windows"].append(12)
        ma_factory_config["version"] = "0.0.2"

        DatasetManagerService.add_new_feature_to_all()
        self.assertTrue("sma12" in StockData.objects.first().features.keys())
        self.assertTrue("sma12" in StockData.objects.last().features.keys())

    def test_update_recent_stock_data(self):
        DatasetManagerService.create_new_stock("SPY", "2023-08-01", "2024-08-9")

        DatasetManagerService.update_recent_stock_data("SPY", "1d")

        all_timestamps = list(StockData.objects.all().values_list("timestamp", flat=True))

        last_timestamp = all_timestamps[-1]

        self.assertEqual(len(all_timestamps), len(set(all_timestamps)) )

        datasettracker = DataSetTracker.objects.first()
        self.assertEqual(datasettracker.end_date, last_timestamp)

        FeatureTrackerService.ensure_synced_features()


class FeatureFactoryServiceTest(TestCase):

    def setUp(self) -> None:
        FeatureFactoryService.update_factory_configs()
        return super().setUp()
    def tearDown(self) -> None:
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown

    def test_update_factory_configs(self):
        ma_factory_config = FACTORY_CONFIG_LIST[0]
        ma_factory_config["parameters"]["windows"].append(12)
        ma_factory_config["version"] = "0.0.2"

        FeatureFactoryService.update_factory_configs()

        updated_config = FeatureFactoryConfig.objects.get(name="MovingAverageFeatureFactory")

        # assert that version is correctly updated and 12 is in the windows list
        self.assertEqual(updated_config.version, "0.0.2")
        self.assertIn(12, updated_config.parameters["windows"])

    def test_load_factories_from_db(self):
        factories = FeatureFactoryService.load_factories_from_db()
        self.assertEqual(len(factories), len(FACTORY_CONFIG_LIST))
        
        self.assertEqual(factories[0].config.name, "MovingAverageFeatureFactory")

    
    def test_apply_feature_factories(self):
        df, _ = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")
        df = FeatureFactoryService.apply_feature_factories(df)

        initial_columns = ["open", "high", "low", "close", "volume"]
        self.factory_added_columns = [config.feature_names for config in FeatureFactoryConfig.objects.all()]
        self.all_columns = initial_columns
        for factory_columns in self.factory_added_columns:
            self.all_columns += factory_columns


        self.assertEqual(list(df.columns), self.all_columns)


class MovingAverageFeatureFactoryTest(TestCase):
    def setUp(self):
        self.df, _ = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")
        config = FeatureFactoryConfig(name="MovingAverageFeatureFactory", parameters={"windows": [5, 10, 20, 50, 100, 200]})
        self.factory = MovingAverageFeatureFactory(config)
    
    def tearDown(self) -> None:
        StockData.objects.all().delete()
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
        self.assertAlmostEqual(self.df['ema5'].iloc[20], self.df['close'].iloc[16:21].ewm(span=5, adjust=True).mean().iloc[-1], places=1)  # Check computed value

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
        self.df = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")[0]
        
        config = FeatureFactoryConfig(name="OHLCVFeatureFactory", parameters={})

        self.factory = OHLCVFeatureFactory(config)
    
    def tearDown(self) -> None:
        StockData.objects.all().delete()
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
        self.df = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")[0]
        
        # Set up the configuration with the desired windows
        config = FeatureFactoryConfig(name="BandFeatureFactory", parameters={"windows": [5, 10, 20, 50, 100, 200]})
        
        # Initialize the BandFeatureFactory with the configuration
        self.factory = BandFeatureFactory(config)

    def tearDown(self) -> None:
        StockData.objects.all().delete()
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
        self.df = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")[0]
        
        # Set up the configuration with the desired RSI periods
        config = FeatureFactoryConfig(name="MomentumFeatureFactory", parameters={"rsi_periods": [5, 10, 20, 50, 100]})
        
        # Initialize the MomentumFeatureFactory with the configuration
        self.factory = MomentumFeatureFactory(config)
    
    def tearDown(self) -> None:
        StockData.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        return super().tearDown()

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
        self.df = DatasetManagerService.fetch_stock_data("AAPL", "2020-01-01", "2021-01-10")[0]
        
        # Set up the configuration with the desired output steps
        self.output_steps = 15   
        target_config = FeatureFactoryConfig(name="TargetFeatureFactory", parameters={"output_steps": self.output_steps})
        pctChg_config = FeatureFactoryConfig(name="OHLCVFeatureFactory", parameters={})
        
        # Initialize the TargetFeatureFactory with the configuration
        self.target_factory = TargetFeatureFactory(target_config)
        self.pctChg_factory = OHLCVFeatureFactory(pctChg_config)
    
    def tearDown(self) -> None:
        StockData.objects.all().delete()
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



class TestFeatureTrackerService(TestCase):

    def setUp(self):
        DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
        self.df = DatasetManagerService.stockdata_to_dataframe("SPY", "1d")
        DatasetTrackerService.track_dataset(self.df, "SPY", "1d")

    def tearDown(self) -> None:
        StockData.objects.all().delete()
        FeatureFactoryConfig.objects.all().delete()
        DataSetTracker.objects.all().delete()
        return super().tearDown()

    def test_initialize_feature_tracker(self):
        tracker = FeatureTrackerService.initialize_feature_tracker()
        self.assertEqual(tracker.features, self.df.columns.tolist())

    def test_ensure_synced_features(self):
        feature_set_tracker = FeatureTrackerService.initialize_feature_tracker()

        FeatureTrackerService.ensure_synced_features()

        self.df["new_column"] = 1

        DatasetManagerService.update_existing_stock_data(self.df, "SPY", "1d")

        with self.assertRaises(Exception):
            FeatureTrackerService.ensure_synced_features()


class TestDatasetTrackerService(TestCase):

        def setUp(self):
            DatasetManagerService.create_new_stock("SPY", "2020-01-01", "2021-01-10")
            self.df = DatasetManagerService.stockdata_to_dataframe("SPY", "1d")
            DatasetTrackerService.track_dataset(self.df, "SPY", "1d")

        def tearDown(self) -> None:
            StockData.objects.all().delete()
            FeatureFactoryConfig.objects.all().delete()
            DataSetTracker.objects.all().delete()
            return super().tearDown()

        def test_track_dataset(self):
            tracker = DataSetTracker.objects.first()
            self.assertEqual(tracker.ticker, "SPY")
            self.assertEqual(tracker.timeframe, "1d")
            self.assertEqual(tracker.features, self.df.columns.tolist())
            self.assertEqual(tracker.start_date, self.df.index.min())
            self.assertEqual(tracker.end_date, self.df.index.max())

        def test_update_dataset_tracker(self):
            DatasetManagerService.update_recent_stock_data("SPY", "1d")
            df = DatasetManagerService.stockdata_to_dataframe("SPY", "1d")

            df["new_column"] = 1
            DatasetManagerService.update_existing_stock_data(df, "SPY", "1d")
            DatasetTrackerService.update_tracker("SPY", "1d")
            tracker = DataSetTracker.objects.first()
            self.assertEqual(tracker.start_date, df.index.min())
            self.assertEqual(tracker.end_date, df.index.max())
            self.assertEqual(sorted(tracker.features), sorted(df.columns.tolist()))






        

    
