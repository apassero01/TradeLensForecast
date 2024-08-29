from abc import ABC, abstractmethod
import pandas as pd 
import pandas_ta as ta
from .models import FeatureFactoryConfig
import numpy as np
import ta as technical_analysis


class FeatureFactory(ABC):
    def __init__(self, config: FeatureFactoryConfig):
        self.config = config
    
    @abstractmethod
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def create_feature_names(self, df: pd.DataFrame) -> pd.DataFrame:
        old_columns = list(df.columns)

        df = self.add_features(df)
        new_columns = list(df.columns)

        self.config.feature_names = [col for col in new_columns if col not in old_columns]
        return df


class OHLCVFeatureFactory(FeatureFactory):

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createPctChg(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createIntraDay(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")

        # fill inf values with backfill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill()
        return df
    
    def createPctChg(self,df):
        '''
        Method to create percent change feature for a stock. 
        '''
        pctChgdf = pd.DataFrame(index = df.index)
        features = ['open', 'high', 'low', 'close', 'volume']
        for feature in features:
            feature_name = 'pctChg' + feature
            pctChgdf[feature_name] = df[feature].pct_change() * 100
            pctChgdf[feature_name] = pctChgdf[feature_name].astype('float64')
            pctChgdf[feature_name] = pctChgdf[feature_name].bfill()


        df.update(pctChgdf)
        new_columns = pctChgdf.loc[:, ~pctChgdf.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createIntraDay(self,df):
        ohlcIntraDay_df = pd.DataFrame(index = df.index)

        ohlcIntraDay_df['opHi'] = (df.high - df.open) / df.open * 100.0

        # % drop from open to low
        ohlcIntraDay_df['opLo'] = (df.low - df.open) / df.open * 100.0

        # % drop from high to close
        ohlcIntraDay_df['hiCl'] = (df.close - df.high) / df.high * 100.0

        # % raise from low to close
        ohlcIntraDay_df['loCl'] = (df.close - df.low) / df.low * 100.0

        # % spread from low to high
        ohlcIntraDay_df['hiLo'] = (df.high - df.low) / df.low * 100.0

        # % spread from open to close
        ohlcIntraDay_df['opCl'] = (df.close - df.open) / df.open * 100.0

        # Calculations for the percentage changes
        ohlcIntraDay_df["pctChgClOp"] = (df.open - df.close.shift(1)) / df.close.shift(1) * 100.0

        ohlcIntraDay_df["pctChgClLo"] = (df.low - df.close.shift(1)) / df.close.shift(1) * 100.0

        ohlcIntraDay_df["pctChgClHi"] = (df.high - df.close.shift(1)) / df.close.shift(1) * 100.0


        for col in ohlcIntraDay_df.columns:
            ohlcIntraDay_df[col] = ohlcIntraDay_df[col].astype('float64')
            ohlcIntraDay_df[col] = ohlcIntraDay_df[col].bfill()
        
        df.update(ohlcIntraDay_df)
        new_columns = ohlcIntraDay_df.loc[:, ~ohlcIntraDay_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
        

class MovingAverageFeatureFactory(FeatureFactory):
    def __init__(self, config: FeatureFactoryConfig):
        super().__init__(config)
        self.windows = self.config.parameters.get("windows", [5, 10, 20])

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.create_SMA(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.create_EMA(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.create_SMA_Volume(df)

        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createSMAPctDiff(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createEMAPctDiff(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createSMAPctDiffVolume(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createSMADerivative(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createEMADerivative(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createSMADerivativeVolume(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill()
        return df
    
    def create_SMA(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create simple moving average feature for a stock. 
        '''
        sma_df = pd.DataFrame(index = df.index)
        for window in self.windows:
            feature_name = "sma" + str(window)
            sma_df[feature_name] = ta.sma(df.close, length=window)
            sma_df[feature_name] = sma_df[feature_name].astype('float64')
            sma_df[feature_name] = sma_df[feature_name].bfill()
        
        df.update(sma_df)
        new_columns = sma_df.loc[:, ~sma_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        return df
    
    def create_EMA(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create exponential moving average feature for a stock. 
        
        '''
        ema_df = pd.DataFrame(index = df.index)
        for window in self.windows:
            feature_name = "ema" + str(window)
            ema_df[feature_name] = ta.ema(df.close, length=window)
            ema_df[feature_name] = ema_df[feature_name].astype('float64')
            ema_df[feature_name] = ema_df[feature_name].bfill()

        df.update(ema_df)
        new_columns = ema_df.loc[:, ~ema_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)  
        return df
    
    def create_SMA_Volume(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create exponential moving average feature for a stocks volume. 
        '''
        sma_vol_df = pd.DataFrame(index = df.index)
        for window in self.windows:
            feature_name = "smaVol" + str(window)
            sma_vol_df[feature_name] = ta.sma(df.volume, length=window)
            sma_vol_df[feature_name] = sma_vol_df[feature_name].astype('float64')
            sma_vol_df[feature_name] = sma_vol_df[feature_name].bfill()
        
        df.update(sma_vol_df)
        new_columns = sma_vol_df.loc[:, ~sma_vol_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createSMAPctDiff(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create percent difference between close price and simple moving averages
        sma_cols = ["smas" + str(window) for window in self.windows]
        '''
        sma_cols = ["sma" + str(window) for window in self.windows]
        sma_pct_diff_df = pd.DataFrame(index = df.index)
        for sma_col in sma_cols:
            feature_name = 'pctDiff+' + sma_col + '_close'
            if not df.index.is_unique:
                print("Warning: Duplicate indices detected.")
            sma_pct_diff_df[feature_name] = (df.close - df[sma_col]) / df[sma_col] * 100
            sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].astype('float64')

            sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].bfill()

            for i, sma_col2 in enumerate(sma_cols):
                if sma_col2 == sma_col:
                    continue
                feature_name = 'pctDiff+' + sma_col + '_' + sma_col2
                sma_pct_diff_df[feature_name] = (df[sma_col] - df[sma_col2]) / df[sma_col2] * 100
                sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].astype('float64')
                sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].bfill()
        
        df.update(sma_pct_diff_df)
        new_columns = sma_pct_diff_df.loc[:, ~sma_pct_diff_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createEMAPctDiff(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create percent difference between close price and exponential moving averages
        '''
        ema_cols = ["ema" + str(window) for window in self.windows]

        ema_pct_diff_df = pd.DataFrame(index = df.index)

        for ema_col in ema_cols:
            feature_name = 'pctDiff+' + ema_col + '_close'
            ema_pct_diff_df[feature_name] = (df.close - df[ema_col]) / df[ema_col] * 100
            ema_pct_diff_df[feature_name] = ema_pct_diff_df[feature_name].astype('float64')

            ema_pct_diff_df[feature_name] = ema_pct_diff_df[feature_name].bfill()

            for i, ema_col2 in enumerate(ema_cols):
                if ema_col2 == ema_col:
                    continue
                feature_name = 'pctDiff+' + ema_col + '_' + ema_col2
                ema_pct_diff_df[feature_name] = (df[ema_col] - df[ema_col2]) / df[ema_col2] * 100
                ema_pct_diff_df[feature_name] = ema_pct_diff_df[feature_name].astype('float64')
                ema_pct_diff_df[feature_name] = ema_pct_diff_df[feature_name].bfill()
            
        df.update(ema_pct_diff_df)
        new_columns = ema_pct_diff_df.loc[:, ~ema_pct_diff_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createSMAPctDiffVolume(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create percent difference between volume and simple moving averages
        '''
        sma_cols = ["smaVol" + str(window) for window in self.windows]

        sma_pct_diff_df = pd.DataFrame(index = df.index)   


        for sma_col in sma_cols:
            feature_name = 'pctDiff+' + sma_col + '_volume'
            sma_pct_diff_df[feature_name] = (df['volume'] - df[sma_col]) / df[sma_col] * 100
            sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].astype('float64')
            sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].bfill()

            for i, sma_col2 in enumerate(sma_cols):
                if sma_col2 == sma_col:
                    continue
                feature_name = 'pctDiff+' + sma_col + '_' + sma_col2
                sma_pct_diff_df[feature_name] = (df[sma_col] - df[sma_col2]) / df[sma_col2] * 100
                sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].astype('float64')
                sma_pct_diff_df[feature_name] = sma_pct_diff_df[feature_name].bfill()

        df.update(sma_pct_diff_df)
        new_columns = sma_pct_diff_df.loc[:, ~sma_pct_diff_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        return df
    
    def createSMADerivative(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create derivative of simple moving averages
        '''
        sma_cols = ["sma" + str(window) for window in self.windows]
        sma_deriv_df = pd.DataFrame(index = df.index)

        for col in sma_cols: 
            period = int(col.replace("sma", ""))
            new_col = "deriv+" + col

            sma_deriv_df[new_col] = (df[col] - df[col].shift(period)) / period
            sma_deriv_df[new_col] = sma_deriv_df[new_col].astype('float64')
            sma_deriv_df[new_col] = sma_deriv_df[new_col].bfill()

        df.update(sma_deriv_df)
        new_columns = sma_deriv_df.loc[:, ~sma_deriv_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        return df
    
    def createEMADerivative(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create derivative of exponential moving averages
        '''
        ema_cols = ["ema" + str(window) for window in self.windows]

        ema_deriv_df = pd.DataFrame(index = df.index)

        for col in ema_cols: 
            period = int(col.replace("ema", ""))
            new_col = "deriv+" + col

            ema_deriv_df[new_col] = (df[col] - df[col].shift(period)) / period
            ema_deriv_df[new_col] = ema_deriv_df[new_col].astype('float64')
            ema_deriv_df[new_col] = ema_deriv_df[new_col].bfill()

        df.update(ema_deriv_df)
        new_columns = ema_deriv_df.loc[:, ~ema_deriv_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        return df
    
    def createSMADerivativeVolume(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Method to create derivative of simple moving averages of volume
        '''
        sma_cols = ["smaVol" + str(window) for window in self.windows]

        sma_deriv_df = pd.DataFrame(index = df.index)

        for col in sma_cols: 
            period = int(col.replace("smaVol", ""))
            new_col = "deriv+" + col
            
            sma_deriv_df[new_col] = (df[col] - df[col].shift(period)) / period
            sma_deriv_df[new_col] = sma_deriv_df[new_col].astype('float64')
            sma_deriv_df[new_col] = sma_deriv_df[new_col].bfill()

        df.update(sma_deriv_df)
        new_columns = sma_deriv_df.loc[:, ~sma_deriv_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        return df
    

class BandFeatureFactory(FeatureFactory):
    '''
    Factory class class to create band related features for a stock
    '''
    def __init__(self, config: FeatureFactoryConfig):
        super().__init__(config)
        self.windows = self.config.parameters.get("windows", [5, 10, 20])
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createBB(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createBBPctDiff(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill()
        return df

    def createBB(self, df):
        bb_df = pd.DataFrame(index = df.index)

        for window in self.windows:
            high_col = "bb_high" + str(window)
            low_col = "bb_low" + str(window)

            bollinger_obj = technical_analysis.volatility.BollingerBands(df.close, window=window, window_dev=2)

            bb_df[high_col] = bollinger_obj.bollinger_hband()
            bb_df[low_col] = bollinger_obj.bollinger_lband()

            bb_df[high_col] = bb_df[high_col].astype('float64')
            bb_df[low_col] = bb_df[low_col].astype('float64')

            bb_df[high_col] = bb_df[high_col].bfill()
            bb_df[low_col] = bb_df[low_col].bfill()

        df.update(bb_df)
        new_columns = bb_df.loc[:, ~bb_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createBBPctDiff(self, df):

        bb_df = pd.DataFrame(index = df.index)

        for window in self.windows:
            high_col = "bb_high" + str(window)
            low_col = "bb_low" + str(window)

            pct_diff_high_low_col = "pctDiff+bb_high_low" + str(window)
            pct_diff_high_close_col = "pctDiff+bb_high_close" + str(window)
            pct_diff_low_close_col = "pctDiff+bb_low_close" + str(window)

            bb_indicator_col = "bb_indicator" + str(window)

            bb_df[pct_diff_high_low_col] = (df[high_col] - df[low_col]) / df[low_col] * 100
            bb_df[pct_diff_high_close_col] = (df[high_col] - df.close) / df.close * 100
            bb_df[pct_diff_low_close_col] = (df[low_col] - df.close) / df.close * 100
            bb_df[bb_indicator_col] = (df.close - df[low_col])/(df[high_col] - df[low_col]) * 100

            bb_df[pct_diff_high_low_col] = bb_df[pct_diff_high_low_col].astype('float64')
            bb_df[pct_diff_high_close_col] = bb_df[pct_diff_high_close_col].astype('float64')
            bb_df[pct_diff_low_close_col] = bb_df[pct_diff_low_close_col].astype('float64')
            bb_df[bb_indicator_col] = bb_df[bb_indicator_col].astype('float64')

            bb_df[pct_diff_high_low_col] = bb_df[pct_diff_high_low_col].bfill()
            bb_df[pct_diff_high_close_col] = bb_df[pct_diff_high_close_col].bfill()
            bb_df[pct_diff_low_close_col] = bb_df[pct_diff_low_close_col].bfill()
            bb_df[bb_indicator_col] = bb_df[bb_indicator_col].bfill()

        df.update(bb_df)
        new_columns = bb_df.loc[:, ~bb_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df


class MomentumFeatureFactory(FeatureFactory):
    '''
    Factory class to create momentum related features for a stock
    '''

    def __init__(self, config: FeatureFactoryConfig):
        super().__init__(config)
        self.rsi_periods = self.config.parameters.get("rsi_periods", [5, 10, 20, 50, 100])
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createRSI(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createMACD(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.createStock(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill()
        return df

    def createRSI(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi_df = pd.DataFrame(index = df.index)

        for period in self.rsi_periods:
            rsi_col = "rsi" + str(period)

            rsi = ta.rsi(df.close, length=period)
            rsi_df[rsi_col] = rsi

            rsi_df[rsi_col] = rsi_df[rsi_col].astype('float64')
            rsi_df[rsi_col] = rsi_df[rsi_col].bfill()

        df.update(rsi_df)
        new_columns = rsi_df.loc[:, ~rsi_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createMACD(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialize the MACD object
        macd = technical_analysis.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
        
        # Create a DataFrame to hold the MACD components
        macd_df = pd.DataFrame(index=df.index)
        
        # Calculate MACD components
        macd_df["macd"] = macd.macd()
        macd_df["macd_signal"] = macd.macd_signal()
        macd_df["macd_diff"] = macd.macd_diff()

        # Ensure correct data types
        macd_df["macd"] = macd_df["macd"].astype('float64')
        macd_df["macd_signal"] = macd_df["macd_signal"].astype('float64')
        macd_df["macd_diff"] = macd_df["macd_diff"].astype('float64')

        # Fill missing values
        macd_df["macd"] = macd_df["macd"].bfill()
        macd_df["macd_signal"] = macd_df["macd_signal"].bfill()
        macd_df["macd_diff"] = macd_df["macd_diff"].bfill()

        # Update the original DataFrame with MACD components
        df.update(macd_df)
        
        # Add the new columns to the original DataFrame
        new_columns = macd_df.loc[:, ~macd_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df
    
    def createStock(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialize the Stochastic Oscillator object
        stoch = technical_analysis.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3, fillna=False)
        
        # Create a DataFrame to hold the Stochastic Oscillator components
        stock_df = pd.DataFrame(index=df.index)
        
        # Calculate Stochastic Oscillator components
        stock_df["stock_k"] = stoch.stoch()
        stock_df["stock_d"] = stoch.stoch_signal()

        # Ensure correct data types
        stock_df["stock_k"] = stock_df["stock_k"].astype('float64')
        stock_df["stock_d"] = stock_df["stock_d"].astype('float64')

        # Fill missing values
        stock_df["stock_k"] = stock_df["stock_k"].bfill()
        stock_df["stock_d"] = stock_df["stock_d"].bfill()

        # Update the original DataFrame with Stochastic Oscillator components
        df.update(stock_df)
        
        # Add the new columns to the original DataFrame
        new_columns = stock_df.loc[:, ~stock_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)

        return df


class TargetFeatureFactory(FeatureFactory):
    '''
    Factory class to create target related features for a stock
    '''
    def __init__(self, config: FeatureFactoryConfig):
        super().__init__(config)
        self.output_steps = self.config.parameters.get("output_steps", 1)
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        df = self.create_pct_chg_target(df)
        if not df.index.is_unique:
            print("Warning: Duplicate indices detected before processing.")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.bfill()
        
        return df
    
    def create_pct_chg_target(self, df):
        '''
        Method to create percent change target for a stock. 
        '''
        target_df = pd.DataFrame(index = df.index)

        for lag in range(1, self.output_steps + 1):
            target_name = 'pctChgclose+' + str(lag)
            shifted_series = df['pctChgclose'].shift(-lag)
            # print length of index in shifted_series and df
            
            target_df[target_name] = shifted_series
            target_df[target_name] = target_df[target_name].astype('float64')
            target_df[target_name] = target_df[target_name].fillna(-999)
        
        cols = list(target_df.columns)
        cols.reverse()
        target_df = target_df[cols]

        df.update(target_df)
        new_columns = target_df.loc[:, ~target_df.columns.isin(df.columns)]
        df = pd.concat([df, new_columns], axis=1)
        
        return df
    

    
        