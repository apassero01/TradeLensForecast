from django.test import TestCase
import numpy as np
from shared_utils.utils.DataUtils import remove_nan_rows
from training_session.VizProcessingStrategies import LineVizProcessingStrategy, SequenceVizProcessingStrategy


class LineVizProcessingStrategyTests(TestCase):

    def setUp(self):
        # Base configurations for the test cases with all necessary default values
        self.config_X_mean = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': LineVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'LineVizProcessingStrategy',
            'feature_type': 'X',
            'graph_type': 'line',
            'aggregation_func': 'mean',
            'data_selection': {
                'model_set_X_train': True,
                'model_set_X_test': False,
                'model_set_y_train': False,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'X_features': ['Feature1', 'Feature2'],
            'y_features': None
        }

        self.config_Y_mean = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': LineVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'LineVizProcessingStrategy',
            'feature_type': 'y',
            'graph_type': 'line',
            'aggregation_func': 'mean',
            'data_selection': {
                'model_set_X_train': False,
                'model_set_X_test': False,
                'model_set_y_train': True,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'X_features': None,
            'y_features': ['Y_feature']
        }

        self.config_X_cluster = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': LineVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'LineVizProcessingStrategy',
            'feature_type': 'X',
            'graph_type': 'line',
            'aggregation_func': 'cluster',
            'data_selection': {
                'model_set_X_train': True,
                'model_set_X_test': False,
                'model_set_y_train': False,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'X_features': ['Feature1', 'Feature2'],
            'y_features': None
        }

        self.config_Y_cluster = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': LineVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'LineVizProcessingStrategy',
            'feature_type': 'y',
            'graph_type': 'line',
            'aggregation_func': 'cluster',
            'data_selection': {
                'model_set_X_train': False,
                'model_set_X_test': False,
                'model_set_y_train': True,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'X_features': None,
            'y_features': ['Y_feature']
        }

        # Sample mock data for testing
        self.data_X = np.random.rand(10, 5, 2)  # (elements, time steps, features)
        self.data_Y = np.random.rand(10, 5, 1)  # (elements, time steps, 1)

        # Initialize strategies
        self.strategy_X_mean = LineVizProcessingStrategy(self.config_X_mean)
        self.strategy_Y_mean = LineVizProcessingStrategy(self.config_Y_mean)
        self.strategy_X_cluster = LineVizProcessingStrategy(self.config_X_cluster)
        self.strategy_Y_cluster = LineVizProcessingStrategy(self.config_Y_cluster)

    def test_mean_aggregation_X(self):
        """Test mean aggregation for X features with additional dimension."""
        result = self.strategy_X_mean.apply(self.data_X)

        # Check output structure and shape for each feature
        for feature in self.config_X_mean['X_features']:
            self.assertIn(feature, result)
            self.assertEqual(len(result[feature]), 1)  # Extra dimension
            self.assertEqual(len(result[feature][0]), self.data_X.shape[1])

    def test_mean_aggregation_Y(self):
        """Test mean aggregation for Y features with additional dimension for consistency."""
        result = self.strategy_Y_mean.apply(self.data_Y)

        # Check output structure and shape for the Y feature
        self.assertIn('Y_feature', result)
        self.assertEqual(len(result['Y_feature']), 1)  # Extra dimension for consistency
        self.assertEqual(len(result['Y_feature'][0]), self.data_Y.shape[1])  # Check time steps

    def test_cluster_aggregation_X(self):
        """Test cluster aggregation for X features with multiple clusters along the first dimension."""
        result = self.strategy_X_cluster.apply(self.data_X)

        # Check output structure and shape for clustering
        for feature in self.config_X_cluster['X_features']:
            self.assertIn(feature, result)
            self.assertEqual(len(result[feature]), 3)  # Number of clusters
            self.assertEqual(len(result[feature][0]), self.data_X.shape[1])

    def test_cluster_aggregation_Y(self):
        """Test cluster aggregation for Y features using TSlearn's TimeSeriesKMeans."""
        result = self.strategy_Y_cluster.apply(self.data_Y)

        # Check that 'Y_feature' is in the result dictionary
        self.assertIn('Y_feature', result)

        # Verify that there are 3 clusters
        self.assertEqual(len(result['Y_feature']), 3)  # Number of clusters

        # Verify that each cluster's time series has the expected number of time steps
        for cluster_time_series in result['Y_feature']:
            self.assertEqual(len(cluster_time_series), self.data_Y.shape[1])

    def test_invalid_feature_type(self):
        """Test invalid feature type raises an error."""
        config_invalid = {
            'feature_type': 'Z',  # Invalid feature type
            'X_features': ['Feature1'],
            'aggregation_func': 'mean'
        }
        with self.assertRaises(ValueError):
            LineVizProcessingStrategy(config_invalid)

    def test_invalid_aggregation_function(self):
        """Test invalid aggregation function raises an error."""
        config_invalid = {
            'feature_type': 'X',
            'X_features': ['Feature1'],
            'aggregation_func': 'invalid_func'
        }
        with self.assertRaises(ValueError):
            LineVizProcessingStrategy(config_invalid)



class SequenceVizProcessingStrategyTests(TestCase):

    def setUp(self):
        # Base configuration for the test cases with necessary default values
        self.config_X = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': SequenceVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'SequenceVizProcessingStrategy',
            'feature_type': 'X',
            'graph_type': 'multiline',
            'data_selection': {
                'model_set_X_train': True,
                'model_set_X_test': False,
                'model_set_y_train': False,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'X_features': ['Feature1', 'Feature2'],
            'index_list': [0, 1, 2],
            'data_set_metadata': []  # Will be populated in setUp
        }

        self.config_y = {
            'parent_strategy': 'VizProcessingStrategy',
            'name': SequenceVizProcessingStrategy.__name__,
            'm_service': 'training_session',
            'type': 'SequenceVizProcessingStrategy',
            'feature_type': 'y',
            'graph_type': 'multiline',
            'data_selection': {
                'model_set_X_train': False,
                'model_set_X_test': False,
                'model_set_y_train': True,
                'model_set_y_test': False,
                'session_X_train': False,
                'session_X_test': False,
                'session_y_train': False,
                'session_y_test': False
            },
            'y_features': ['Y_feature'],
            'index_list': [0, 1, 2],
            'data_set_metadata': []  # Will be populated in setUp
        }

        # Sample mock data for testing
        self.data_X = np.random.rand(3, 5, 2)  # (elements, time steps, features)
        self.data_y = np.random.rand(3, 5, 1)  # (elements, time steps, 1)

        # Generate fake metadata
        self.data_set_metadata = []
        for i in range(3):
            meta_data = {}
            meta_data['id'] = i
            meta_data['start_timestamp'] = '2020-01-01T00:00:00Z'
            meta_data['end_timestamp'] = '2020-01-01T01:00:00Z'
            meta_data['metadata'] = {'ticker': f'TICKER{i}'}
            self.data_set_metadata.append(meta_data)

        # Update configs with metadata
        self.config_X['data_set_metadata'] = self.data_set_metadata
        self.config_y['data_set_metadata'] = self.data_set_metadata

        # Initialize strategies
        self.strategy_X = SequenceVizProcessingStrategy(self.config_X)
        self.strategy_y = SequenceVizProcessingStrategy(self.config_y)

    def test_apply_X(self):
        """Test apply_X method for X features and verify the structure and metadata."""
        result = self.strategy_X.apply_X(self.data_X)

        # Expected number of sequences: elements
        expected_length = self.data_X.shape[0]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), expected_length)

        feature_list = self.config_X['X_features']

        # Iterate over the result and check the structure
        for i, item in enumerate(result):
            self.assertIsInstance(item, dict)
            self.assertEqual(len(item), len(feature_list))  # Should have one key for each feature
            for feature in feature_list:
                self.assertIn(feature, item)  # Feature name should be a key
                data_dict = item[feature]
                self.assertIn('data', data_dict)
                self.assertIn('metadata', data_dict)
                self.assertIsInstance(data_dict['data'], list)
                self.assertEqual(len(data_dict['data']), 1)  # Single sequence (axis=0 dimension added)
                self.assertEqual(len(data_dict['data'][0]), self.data_X.shape[1])  # Time steps

                # Check metadata
                meta = data_dict['metadata']
                self.assertIn('id', meta)
                self.assertIn('start_timestamp', meta)
                self.assertIn('end_timestamp', meta)
                self.assertIn('metadata', meta)
                self.assertIn('ticker', meta['metadata'])

    def test_apply_Y(self):
        """Test apply_Y method for Y features and verify the structure and metadata."""
        result = self.strategy_y.apply_Y(self.data_y)

        # Expected number of sequences: elements (since Y has a single feature)
        expected_length = self.data_y.shape[0]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), expected_length)

        feature_name = self.config_y['y_features'][0]

        # Iterate over the result and check the structure
        for i, item in enumerate(result):
            self.assertIsInstance(item, dict)
            self.assertEqual(len(item), 1)  # Should have only one key (the feature name)
            self.assertIn(feature_name, item)  # Feature name should be a key
            data_dict = item[feature_name]
            self.assertIn('data', data_dict)
            self.assertIn('metadata', data_dict)
            self.assertIsInstance(data_dict['data'], list)
            self.assertEqual(len(data_dict['data']), 1)  # Single sequence (axis=0 dimension added)
            self.assertEqual(len(data_dict['data'][0]), self.data_y.shape[1])  # Time steps

            # Check metadata
            meta = data_dict['metadata']
            self.assertIn('id', meta)
            self.assertIn('start_timestamp', meta)
            self.assertIn('end_timestamp', meta)
            self.assertIn('metadata', meta)
            self.assertIn('ticker', meta['metadata'])


    def test_missing_feature_type(self):
        """Test that missing feature_type in config raises ValueError."""
        config_invalid = self.config_X.copy()
        del config_invalid['feature_type']
        with self.assertRaises(ValueError):
            SequenceVizProcessingStrategy(config_invalid)

    def test_invalid_feature_type(self):
        """Test that invalid feature_type in config raises ValueError."""
        config_invalid = self.config_X.copy()
        config_invalid['feature_type'] = 'InvalidType'
        with self.assertRaises(ValueError):
            SequenceVizProcessingStrategy(config_invalid)

    def test_missing_index_list(self):
        """Test that missing index_list in config raises ValueError."""
        config_invalid = self.config_X.copy()
        del config_invalid['index_list']
        with self.assertRaises(ValueError):
            SequenceVizProcessingStrategy(config_invalid)

    def test_missing_data_set_metadata(self):
        """Test that missing data_set_metadata in config raises ValueError."""
        config_invalid = self.config_X.copy()
        del config_invalid['data_set_metadata']
        with self.assertRaises(ValueError):
            SequenceVizProcessingStrategy(config_invalid)

    def test_input_array_shape_mismatch(self):
        """Test that input array with mismatched shape raises ValueError."""
        data_invalid = np.random.rand(4, 5, 2)  # Should be 3 elements as per data_set_metadata
        with self.assertRaises(ValueError):
            self.strategy_X.apply(data_invalid)

    def test_invalid_input_array(self):
        """Test that invalid input array dimensions raise ValueError."""
        data_invalid = np.random.rand(3, 5)  # Not a 3D array
        with self.assertRaises(ValueError):
            self.strategy_X.apply(data_invalid)