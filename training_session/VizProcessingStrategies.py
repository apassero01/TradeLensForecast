from sympy.codegen.cnodes import static
from tslearn.clustering import TimeSeriesKMeans

from shared_utils.strategies import VizProcessingStrategy
from shared_utils.utils.DataUtils import remove_nan_rows
from sklearn.cluster import KMeans
import numpy as np


class HistVizProcessingStrategy(VizProcessingStrategy):
    name = 'Hist'
    def __init__(self, config):
        super().__init__(config)

        if 'feature_type' not in config.keys():
            raise ValueError("Missing feature_type in config")
        if 'X' != config['feature_type'] and 'y' != config['feature_type']:
            raise ValueError("Invalid feature_type")

        if config['feature_type'] == 'X':
            self.feature_list = config['X_features']
        else:
            self.feature_list = config['y_features']


    def apply(self, arr):

        arr = remove_nan_rows(arr)

        if self.config['feature_type'] == 'X':
            return self.apply_X(arr)
        elif self.config['feature_type'] == 'y':
            return self.apply_y(arr)
        else:
            raise ValueError("Invalid feature_type")

    def apply_X(self, arr):
        """
        Compress a 3D array (shape: [N, M, F]) into a 2D dictionary by merging the first two dimensions.

        Parameters:
            arr (numpy.ndarray): 3D array to be compressed. Expected shape is (N, M, F).

        Returns:
            dict: Dictionary where each key corresponds to a feature in `self.features`
                  and each value is the compressed 2D array data for that feature.
        """
        if arr.ndim != 3:
            raise ValueError("Input array must be a 3D array")

        # Flatten the first two dimensions, converting (N, M, F) -> (N*M, F)
        compressed_data = arr[:,0,:]
        # compressed_data = arr.reshape(-1, arr.shape[-1])

        # Map each feature to the corresponding data column
        result = {self.feature_list[i]: compressed_data[:, i].tolist() for i in range(len(self.feature_list))}

        return result

    def apply_y(self, arr):
        """
        Compress a 3D array (shape: [N, M, F]) into a 2D dictionary by merging the first two dimensions.

        Parameters:
            arr (numpy.ndarray): 3D array to be compressed. Expected shape is (N, M, F).

        Returns:
            dict: Dictionary where each key corresponds to a feature in `self.features`
                  and each value is the compressed 2D array data for that feature.
        """
        if arr.ndim != 3:
            raise ValueError("Input array must be a 3D array")

        # Flatten the first two dimensions, converting (N, M, F) -> (N*M, F)
        compressed_data = arr.squeeze(-1)

        print(compressed_data.shape)

        # Map each feature to the corresponding data column
        result = {self.feature_list[i]: compressed_data[:, i].tolist() for i in range(len(self.feature_list))}

        return result


    @staticmethod
    def get_default_config():
        """
        Returns a default configuration for CompressionStrategy.
        """
        return {
            'parent_strategy': 'VizProcessingStrategy',
            'name': HistVizProcessingStrategy.__name__,
            'm_service': 'training_session',  # Replace with appropriate service name
            'type': 'HistVizProcessingStrategy',
            'feature_type': 'X',  # Default to 'X' for input features
            'graph_type': 'multihist',
            'data_selection': {
                'model_set_X_train' : False,
                'model_set_X_test' : False,
                'model_set_y_train' : False,
                'model_set_y_test' : False,
                'session_X_train' : False,
                'session_X_test' : False,
                'session_y_train' : False,
                'session_y_test' : False

            },
            'X_features': None,
            'y_features': None
        }


class LineVizProcessingStrategy(VizProcessingStrategy):
    name = 'Line'

    def __init__(self, config):
        super().__init__(config)

        # Validate feature type
        if 'feature_type' not in config.keys():
            raise ValueError("Missing feature_type in config")
        if config['feature_type'] not in ['X', 'y']:
            raise ValueError("Invalid feature_type")

        # Select feature list based on type
        if config['feature_type'] == 'X':
            self.feature_list = config['X_features']
            self.is_y_feature = False
        else:
            self.feature_list = ['Y_feature']  # Single feature for Y
            self.is_y_feature = True

        # Retrieve the aggregation function name
        func_name = config.get('aggregation_func', 'mean')
        if func_name not in ['mean', 'cluster']:
            raise ValueError(f"Aggregation function '{func_name}' is not supported.")

        # Set the chosen aggregation function
        self.aggregation_func = func_name

    def apply(self, arr):
        # Remove NaN rows
        arr = remove_nan_rows(arr)

        # Process X or Y based on configuration
        if not self.is_y_feature:
            return self.apply_X(arr)
        else:
            return self.apply_Y(arr)

    def apply_X(self, arr):
        """
        Process the 3D array for X features (samples, time steps, features) and prepare for aggregation.
        """
        if arr.ndim != 3:
            raise ValueError("Input array must be a 3D array")

        # Reshape and aggregate data
        reshaped_data = arr  # No need to reshape, as X is already in (samples, time steps, features)
        return self._aggregate(reshaped_data)

    def apply_Y(self, arr):
        """
        Process the 3D array for Y features (elements, time steps, 1) and reshape for aggregation.

        Parameters:
            arr (numpy.ndarray): Input 3D array with shape (elements, time steps, 1).

        Returns:
            dict: Dictionary with aggregated values.
        """
        if arr.ndim != 3 or arr.shape[-1] != 1:
            raise ValueError("Input array for Y must be a 3D array with shape (elements, time steps, 1)")

        # Reshape Y data to match X's format for aggregation
        reshaped_data = arr.squeeze(-1)  # Shape becomes (elements, time steps)
        reshaped_data = reshaped_data[:, :,
                        np.newaxis]  # Now (elements, time steps, 1) to match (elements, time steps, features)

        return self._aggregate(reshaped_data)

    def _aggregate(self, data):
        """
        General aggregation method that applies the chosen aggregation function on reshaped data.

        Parameters:
            data (numpy.ndarray): 3D array (elements, time steps, features) ready for aggregation.

        Returns:
            dict: Aggregated data, where each feature contains another dimension for multiple items if clustering.
        """
        # Choose and apply the aggregation function
        if self.aggregation_func == 'mean':
            return self._mean_aggregation(data)
        elif self.aggregation_func == 'cluster':
            return self._cluster_aggregation(data)

    def _mean_aggregation(self, data):
        """
        Mean aggregation for data across elements along the time step axis.

        Parameters:
            data (numpy.ndarray): Input data array.

        Returns:
            dict: Dictionary where each feature has mean-aggregated values over time,
                  with an extra dimension for consistency with cluster aggregation.
        """
        # Compute mean along the elements axis (axis=0)
        aggregated_over_time = np.mean(data, axis=0)  # Shape: (time steps, features)

        # Wrap each feature’s mean result in an additional list to add the extra dimension
        result = {
            self.feature_list[i]: [aggregated_over_time[:, i].tolist()]  # Shape: (1, time steps)
            for i in range(len(self.feature_list))
        }
        return result

    def _cluster_aggregation(self, data, n_clusters=3):
        """
        Cluster aggregation using TSlearn's TimeSeriesKMeans for each feature independently.

        Parameters:
            data (numpy.ndarray): Input data array with shape (elements, time steps, features).
            n_clusters (int): Number of clusters for TimeSeriesKMeans.

        Returns:
            dict: Dictionary where each feature has multiple cluster-aggregated values over time.
        """
        print("Clustering data with shape:", data.shape)
        n_elements, n_time_steps, n_features = data.shape

        # Result dictionary to store the clustered time series for each feature
        result = {}

        # Apply clustering to each feature independently
        for i, feature in enumerate(self.feature_list):
            # Extract data for the current feature with shape (elements, time steps)
            feature_data = data[:, :, i]

            # Initialize the TimeSeriesKMeans model for the current feature
            ts_kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=False)

            # Fit TimeSeriesKMeans to the current feature’s data
            ts_kmeans.fit(feature_data)

            # Retrieve the cluster centers; shape is (n_clusters, time_steps)
            cluster_centers = ts_kmeans.cluster_centers_

            # Store the cluster centers for this feature, ensuring correct shape (n_clusters, time_steps)
            result[feature] = cluster_centers.reshape(n_clusters, n_time_steps).tolist()

        return result

    @staticmethod
    def get_default_config():
        """
        Returns a default configuration for CompressionStrategy.
        """
        return {
            'parent_strategy': 'VizProcessingStrategy',
            'name': LineVizProcessingStrategy.__name__,
            'm_service': 'training_session',  # Replace with appropriate service name
            'type': 'LineVizProcessingStrategy',
            'feature_type': 'X',  # Default to 'X' for input features
            'graph_type': 'multiline',
            'data_selection': {
                'model_set_X_train' : False,
                'model_set_X_test' : False,
                'model_set_y_train' : False,
                'model_set_y_test' : False,
                'session_X_train' : False,
                'session_X_test' : False,
                'session_y_train' : False,
                'session_y_test' : False

            },
            'aggregation_func': 'cluster',
            'X_features': None,
            'y_features': None
        }

class SequenceVizProcessingStrategy(VizProcessingStrategy):
    name = 'Sequence'

    def __init__(self, config):
        super().__init__(config)

        # Validate feature type
        if 'feature_type' not in config.keys():
            raise ValueError("Missing feature_type in config")
        if config['feature_type'] not in ['X', 'y']:
            raise ValueError("Invalid feature_type")
        if 'index_list' not in config.keys():
            raise ValueError("Missing index_list in config")
        if 'data_set_metadata' not in config.keys():
            raise ValueError("Missing data_set_metadata in config. Not added by Strategy")

        # Select feature list based on type
        if config['feature_type'] == 'X':
            self.feature_list = config['X_features']
            self.is_y_feature = False
        else:
            self.feature_list = ['Y_feature']  # Single feature for Y
            self.is_y_feature = True

        self.data_set_metadata = config['data_set_metadata']



    def apply(self, arr):
        if arr.shape[0] != len(self.data_set_metadata):
            raise ValueError("Data set metadata length does not match input array")

        if arr.ndim != 3:
            raise ValueError("Input array must be a 3D array")

        arr = remove_nan_rows(arr)
        if not self.is_y_feature:
            return self.apply_X(arr)
        else:
            return self.apply_Y(arr)

    def apply_X(self, arr):
        """
        Process the 3D array for X features (samples, time steps, features)
        Parameters:
            arr (numpy.ndarray): Input 3D array with shape (elements, time steps, 1).

        Returns:
            dict: Dictionary with sequences and metadata

        """

        ret_data = []
        for i in range(arr.shape[0]):
            single_seq = {}
            for j, feature in enumerate(self.feature_list):
                seq_data = arr[i, :, j]
                # add another dimension along axis = 0
                seq_data = seq_data[np.newaxis, :]
                single_seq[feature] = {'data': seq_data.tolist(), 'metadata': self.data_set_metadata[i]}
            ret_data.append(single_seq)
        print(ret_data)
        return ret_data


    def apply_Y(self, arr):
        """
        Process the 3D array for Y features (elements, time steps, 1) and return elements with metadata

        Parameters:
            arr (numpy.ndarray): Input 3D array with shape (elements, time steps, 1).

        Returns:
            dict: Dictionary with sequences and metadata

        """
        ret_data = []
        for i in range(arr.shape[0]):
            seq_data = arr[i, :, 0]
            # add another dimension along axis = 0
            seq_data = seq_data[np.newaxis, :]
            ret_data.append({self.feature_list[0]: {'data': seq_data.tolist(), 'metadata': self.data_set_metadata[i]}})

        return ret_data


    @staticmethod
    def get_default_config():
        """
        Returns a default configuration for CompressionStrategy.
        """
        return {
            'parent_strategy': 'VizProcessingStrategy',
            'name': SequenceVizProcessingStrategy.__name__,
            'm_service': 'training_session',  # Replace with appropriate service name
            'type': 'SequenceVizProcessingStrategy',
            'feature_type': 'X',  # Default to 'X' for input features
            'graph_type': 'sequence_multiline',
            'index_list': [],
            'data_selection': {
                'model_set_X_train' : False,
                'model_set_X_test' : False,
                'model_set_y_train' : False,
                'model_set_y_test' : False,
                'session_X_train' : False,
                'session_X_test' : False,
                'session_y_train' : False,
                'session_y_test' : False

            },
            'X_features': None,
            'y_features': None
        }



