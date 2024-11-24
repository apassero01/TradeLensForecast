from enum import Enum

import numpy as np


class Scaler:
    def fit(self, X):
        pass
    def transform(self, X):
        pass
    def fit_transform(self, X):
        pass
    def inverse_transform(self, X):
        pass
    def serialize(self):
        pass
    def deserialize(self, config):
        pass

class MeanVarianceScaler3D(Scaler):
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

    def serialize(self):
        if self.mean_ is None or self.var_ is None:
            return {}
        return {'mean': self.mean_.tolist(), 'var': self.var_.tolist()}

    def deserialize(self, config):
        self.mean_ = np.array(config['mean'])
        self.var_ = np.array(config['var'])
        return self

class TimeStepScaler3D:
    '''
    This class is used to scale 3D time series data, where each time step is scaled independently.

    I think this is for pctChg y values where we want each output step to be scaled independently
    '''
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.eps = 1e-8  # Small constant to avoid division by zero

    def fit(self, X):
        """
        Fit the scaler to the 3D time series data, calculating a unique mean and variance for each time step.
        Parameters:
        X : numpy array of shape (n_samples, n_timesteps, n_features)
        """
        # Compute mean and variance for each time step across all samples and features
        self.mean_ = np.mean(X, axis=0)  # Shape will be (n_timesteps, 1)
        self.var_ = np.var(X, axis=0)    # Shape will be (n_timesteps, 1)
        return self

    def transform(self, X):
        """
        Transform the 3D time series data using the fitted mean and variance for each time step.
        Parameters:
        X : numpy array of shape (n_samples, n_timesteps, n_features)
        Returns:
        Scaled X of the same shape
        """
        if self.mean_ is None or self.var_ is None:
            raise ValueError(
                "This TimeStepScaler3D instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # Subtract mean and divide by standard deviation for each time step
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
                "This TimeStepScaler3D instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        return X_scaled * np.sqrt(self.var_ + self.eps) + self.mean_

    def serialize(self):
        if self.mean_ is None or self.var_ is None:
            return {}
        return {'mean': self.mean_.tolist(), 'var': self.var_.tolist()}

    def deserialize(self, config):
        self.mean_ = np.array(config['mean'])
        self.var_ = np.array(config['var'])
        return self


class MinMaxSeqBySeqScaler3D(Scaler):
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        # X is (elements, time_steps, features)
        self.min_ = np.min(X, axis=1)  # Shape (elements, features)
        self.max_ = np.max(X, axis=1)  # Shape (elements, features)
        # Avoid division by zero
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1.0

    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_scaled = (X - self.min_[:, np.newaxis, :]) / self.range_[:, np.newaxis, :]
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X = X_scaled * self.range_[:, np.newaxis, :] + self.min_[:, np.newaxis, :]
        return X

    def serialize(self):
        config = {
            'min': self.min_.tolist() if self.min_ is not None else [],
            'max': self.max_.tolist() if self.max_ is not None else [],
        }
        return config

    def deserialize(self, config):
        self.min_ = np.array(config['min']) if config['min'] else None
        self.max_ = np.array(config['max']) if config['max'] else None
        if self.min_ is not None and self.max_ is not None:
            self.range_ = self.max_ - self.min_
            self.range_[self.range_ == 0] = 1.0
        else:
            self.range_ = None

class MaxSeqBySeqScaler3D(Scaler):
    def __init__(self):
        self.max_ = None

    def fit(self, X):
        # X is (elements, time_steps, features)
        self.max_ = np.max(X, axis=(1,2))  # Shape (elements,)
        # Avoid division by zero
        self.max_[self.max_ == 0] = 1.0

    def transform(self, X):
        if self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_scaled = X / self.max_[:, np.newaxis, np.newaxis]
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X = X_scaled * self.max_[:, np.newaxis, np.newaxis]
        return X

    def serialize(self):
        config = {
            'max': self.max_.tolist() if self.max_ is not None else [],
        }
        return config

    def deserialize(self, config):
        self.max_ = np.array(config['max']) if config['max'] else None


class MinMaxSeqBySeqScaler2D(Scaler):
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        # Flatten X to 2D if it's 3D
        X_reshaped = self._reshape_input(X)
        self.min_ = np.min(X_reshaped, axis=1)  # Shape: (samples,)
        self.max_ = np.max(X_reshaped, axis=1)  # Shape: (samples,)
        self.range_ = self.max_ - self.min_
        # Avoid division by zero
        self.range_[self.range_ == 0] = 1.0

    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_reshaped = self._reshape_input(X)
        X_scaled = (X_reshaped - self.min_[:, np.newaxis]) / self.range_[:, np.newaxis]
        return self._reshape_output(X_scaled, X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        X_scaled_reshaped = self._reshape_input(X_scaled)
        X = X_scaled_reshaped * self.range_[:, np.newaxis] + self.min_[:, np.newaxis]
        return self._reshape_output(X, X_scaled)

    def _reshape_input(self, X):
        if X.ndim == 2:
            # Input is already 2D: (samples, features)
            return X
        elif X.ndim == 3:
            # Flatten the last two dimensions
            return X.reshape(X.shape[0], -1)
        else:
            raise ValueError("Input array must be 2D or 3D.")

    def _reshape_output(self, X_reshaped, X_original):
        # Reshape back to the original shape
        return X_reshaped.reshape(X_original.shape)

    def serialize(self):
        config = {
            'min': self.min_.tolist() if self.min_ is not None else [],
            'max': self.max_.tolist() if self.max_ is not None else [],
            'range': self.range_.tolist() if self.range_ is not None else [],
        }
        return config

    def deserialize(self, config):
        self.min_ = np.array(config['min']) if config['min'] else None
        self.max_ = np.array(config['max']) if config['max'] else None
        self.range_ = np.array(config['range']) if config['range'] else None

class ScalerEnum(Enum):
    MEAN_VARIANCE_SCALER_3D = MeanVarianceScaler3D
    TIME_STEP_SCALER_3D = TimeStepScaler3D
    MIN_MAX_SEQ_BY_SEQ_3D = MinMaxSeqBySeqScaler3D
    MAX_SCALER_3D = MaxSeqBySeqScaler3D
    MIN_MAX_SEQ_BY_SEQ_2D = MinMaxSeqBySeqScaler2D

