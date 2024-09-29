

## Feature Factory Configs 

MOVING_AVERAGE_FEATURE_FACTORY_CONFIG = {
    "name": "MovingAverageFeatureFactory",
    "description": "Calculates the Simple Moving Average for a given window size",
    "parameters": {
        "windows": [5, 10, 20, 50, 100, 200],
    },
    "version": "0.0.1",
    "class_path": "dataset_manager.factories.MovingAverageFeatureFactory",

}

OHLCV_FEATURE_FACTORY_CONFIG = {
    "name": "OHLCFeatureFactory",
    "description": "Calculates the Open, High, Low, Close price intra-day differences and pctChg differences",
    "parameters": {},
    "version": "0.0.1",
    "class_path": "dataset_manager.factories.OHLCVFeatureFactory",
}

BAND_FACTORY_CONFIG = {
    "name": "BollingerBandsFeatureFactory",
    "description": "Calculates the Bollinger Bands for a given window size",
    "parameters": {
        "windows": [10, 20, 50, 100, 200],
    },
    "version": "0.0.1",
    "class_path": "dataset_manager.factories.BandFeatureFactory",
}

MOMENTUM_FACTORY_CONFIG = {
    "name": "MomentumFeatureFactory",
    "description": "Calculates the Momentum for a given window size",
    "parameters": {
        "rsi_periods": [5, 10, 20, 50, 100],
    },
    "version": "0.0.1",
    "class_path": "dataset_manager.factories.MomentumFeatureFactory",
}

TARGET_FACTORY_CONFIG = {
    "name": "TargetFeatureFactory",
    "description": "Calculates the target variable for the models",
    "parameters": {
        "output_steps": 15,
    },
    "version": "0.0.5",
    "class_path": "dataset_manager.factories.TargetFeatureFactory",
}

STOCK_FACTORY_CONFIG_LIST = [MOVING_AVERAGE_FEATURE_FACTORY_CONFIG, OHLCV_FEATURE_FACTORY_CONFIG,
                       BAND_FACTORY_CONFIG, MOMENTUM_FACTORY_CONFIG, TARGET_FACTORY_CONFIG]