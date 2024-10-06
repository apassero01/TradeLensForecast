from enum import Enum

import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from training_manager.models import FeatureSet


class FeatureSetService:
    def create_feature_set(self, feature_set_config):
        scaler_config = feature_set_config['scaler_config']
        scaler_name = scaler_config['scaler_name']
        scaler = self.get_scaler(scaler_name)

        features = feature_set_config['feature_list']

        feature_set = FeatureSet.objects.create(feature_list = features,
                                                scaler_config = scaler_config)

        feature_set.scaler = scaler

        return feature_set

    def get_scaler(self, scaler_name):

        if scaler_name not in ScalerEnum.__members__:
            raise ValueError(f"Scaler {scaler_name} not supported")

        scaler_class = ScalerEnum[scaler_name].value
        return scaler_class()




class MeanVarianceScaler3D:
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.eps = 1e-8  # Small constant to avoid division by zero

    def fit(self, X):
        """
        Fit the scaler to the 3D time series data.
        Parameters:
        X : numpy array of shape (n_samples, n_timesteps, n_features)
        """
        # Compute the mean and variance over the samples and time steps, for each feature
        self.mean_ = np.mean(X, axis=(0, 1))
        self.var_ = np.var(X, axis=(0, 1))
        return self

    def transform(self, X):
        """
        Transform the 3D time series data using the fitted mean and variance.
        Parameters:
        X : numpy array of shape (n_samples, n_timesteps, n_features)
        Returns:
        Scaled X of the same shape
        """
        if self.mean_ is None or self.var_ is None:
            raise ValueError(
                "This MeanVarianceScaler3D instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        return (X - self.mean_) / np.sqrt(self.var_ + self.eps)

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Parameters:
        X : numpy array of shape (n_samples, n_timesteps, n_features)
        Returns:
        Scaled X of the same shape
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Inverse the transformation on the scaled data to recover original values.
        Parameters:
        X_scaled : numpy array of shape (n_samples, n_timesteps, n_features)
        Returns:
        Unscaled (original) X of the same shape
        """
        if self.mean_ is None or self.var_ is None:
            raise ValueError(
                "This MeanVarianceScaler3D instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        return X_scaled * np.sqrt(self.var_ + self.eps) + self.mean_


class ScalerEnum(Enum):
    MEAN_VARIANCE_SCALER_3D = MeanVarianceScaler3D


