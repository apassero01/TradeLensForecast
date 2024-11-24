import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.strategy.DataBundleStrategy import CombineDataBundlesStrategy
from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from sequenceset_manager.strategy.SequenceSetStrategy import CombineSeqBundlesStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.strategy.TrainingSessionStrategy import GetSequenceSetsStrategy, GetDataBundleStrategy


class GetSequenceSetsStrategyTestCase(TestCase):
    def setUp(self):
        # Mock features and configurations
        self.X_features = ['open', 'close']
        self.y_features = ['close+1']
        self.dataset_type = 'stock'

        # Strategy request mock (replaces config in the old test)
        self.strategy_request = MagicMock()
        self.strategy_request.param_config = {
            'X_features': self.X_features,
            'y_features': self.y_features,
            'dataset_type': self.dataset_type,
            'model_set_configs': [
                {
                    'sequence_length': 10,
                    'start_timestamp': '2023-01-01',
                    'interval': '1d',
                    'ticker': 'AAPL',
                },
                {
                    'sequence_length': 20,
                    'start_timestamp': '2023-01-01',
                    'interval': '1d',
                    'ticker': 'AAPL',
                },
            ],
        }

        # Mock the TrainingSessionEntity
        self.session_entity = TrainingSessionEntity()
        self.session_entity.X_features = self.X_features
        self.session_entity.y_features = self.y_features

        # Initialize the strategy
        self.strategy_executor = MagicMock()
        self.strategy = GetSequenceSetsStrategy(self.strategy_executor, self.strategy_request)

    @patch('training_session.strategy.TrainingSessionStrategy.requests.get')  # Correct path to patch `requests.get`
    def test_apply(self, mock_get):
        # Mock the HTTP response
        mock_response_data = [
            {
                'id': 1,
                'start_timestamp': '2023-01-01T00:00:00Z',
                'end_timestamp': '2023-01-10T00:00:00Z',
                'sliced_data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
            {
                'id': 2,
                'start_timestamp': '2023-01-01T00:00:00Z',
                'end_timestamp': '2023-01-10T00:00:00Z',
                'sliced_data': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            },
        ]
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response_data

        # Call the apply method
        self.strategy.apply(self.session_entity)

        sequence_sets = self.session_entity.get_entity(EntityEnum.SEQUENCE_SETS)

        # Validate the results
        self.assertEqual(len(sequence_sets), 2)

        first_sequence_set = sequence_sets[0]
        self.assertEqual(first_sequence_set.dataset_type, self.dataset_type)
        self.assertEqual(first_sequence_set.sequence_length, 10)
        self.assertEqual(first_sequence_set.start_timestamp, '2023-01-01')
        self.assertEqual(first_sequence_set.metadata['ticker'], 'AAPL')
        self.assertEqual(first_sequence_set.X_features, self.X_features)
        self.assertEqual(first_sequence_set.y_features, self.y_features)

        first_sequence = first_sequence_set.sequences[0]
        self.assertEqual(first_sequence.sequence_length, 10)
        self.assertEqual(len(first_sequence.sequence_data), 10)
        self.assertTrue(first_sequence.start_timestamp > '2023-01-01')
        self.assertTrue(first_sequence.end_timestamp > first_sequence.start_timestamp)


class GetDataBundleStrategyTestCase(TestCase):
    def setUp(self):
        # Create DataBundles
        self.data_bundle1 = DataBundleEntity()
        self.data_bundle1.set_dataset({
            'X_train': np.array([[1, 2], [3, 4]]),
            'y_train': np.array([1, 0])
        })

        self.data_bundle2 = DataBundleEntity()
        self.data_bundle2.set_dataset({
            'X_train': np.array([[5, 6], [7, 8]]),
            'y_train': np.array([0, 1])
        })

        # Create SequenceSetEntities and populate with DataBundles
        self.sequence_set1 = SequenceSetEntity()
        self.sequence_set1.set_entity_map({EntityEnum.DATA_BUNDLE.value: self.data_bundle1})

        self.sequence_set2 = SequenceSetEntity()
        self.sequence_set2.set_entity_map({EntityEnum.DATA_BUNDLE.value: self.data_bundle2})

        # Create a TrainingSessionEntity and populate with SequenceSets
        self.training_session = TrainingSessionEntity()
        self.training_session.set_entity_map({
            EntityEnum.SEQUENCE_SETS.value: [self.sequence_set1, self.sequence_set2]
        })

        # Create the nested request for CombineSeqBundlesStrategy
        self.nested_strategy_request = StrategyRequestEntity()
        self.nested_strategy_request.strategy_name = CombineSeqBundlesStrategy.__name__
        self.nested_strategy_request.strategy_path = (
            EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SETS.value
        )
        self.nested_strategy_request.param_config = {}

        # Create the main strategy request for GetDataBundleStrategy
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.strategy_name = GetDataBundleStrategy.__name__
        self.strategy_request.strategy_path = 'training_session'
        self.strategy_request.param_config = {}
        self.strategy_request.nested_requests = [self.nested_strategy_request]

        # Register strategies in StrategyExecutor
        self.executor = StrategyExecutor()
        self.executor.register_strategy(CombineSeqBundlesStrategy.__name__, CombineSeqBundlesStrategy)
        self.executor.register_strategy(CombineDataBundlesStrategy.__name__, CombineDataBundlesStrategy)

        # Initialize the strategy
        self.strategy = GetDataBundleStrategy(self.executor, self.strategy_request)

    def test_apply(self):
        # Execute the strategy
        self.strategy.apply(self.training_session)

        # Retrieve the combined data bundle from the session
        combined_data_bundle = self.training_session.get_entity(EntityEnum.DATA_BUNDLE.value)

        # Verify the combined dataset
        expected_X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_y_train = np.array([1, 0, 0, 1])

        np.testing.assert_array_equal(combined_data_bundle.dataset['X_train'], expected_X_train)
        np.testing.assert_array_equal(combined_data_bundle.dataset['y_train'], expected_y_train)

    def test_verify_executable(self):
        # Test that verify_executable does not raise any exceptions
        try:
            self.strategy.verify_executable(self.training_session, self.strategy_request)
        except Exception as e:
            self.fail(f"verify_executable raised an exception: {e}")

    def test_get_request_config(self):
        # Test the static method for request configuration
        request_config = GetDataBundleStrategy.get_request_config()
        self.assertEqual(request_config['strategy_name'], 'GetDataBundleStrategy')
        self.assertEqual(request_config['strategy_path'], 'training_session')
        self.assertIn('nested_requests', request_config)
        self.assertEqual(
            request_config['nested_requests'][0]['strategy_name'], 'CombineSeqBundlesStrategy'
        )