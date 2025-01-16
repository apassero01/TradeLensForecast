import numpy as np
from django.test import TestCase
from unittest.mock import patch, MagicMock

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.strategy.DataBundleStrategy import CombineDataBundlesStrategy
from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.strategy.TrainingSessionStrategy import GetSequenceSetsStrategy


class GetSequenceSetsStrategyTestCase(TestCase):
    def setUp(self):
        # Mock features and configurations
        self.X_features = ['open', 'close']
        self.y_features = ['close+1']
        self.dataset_type = 'stock'

        self.entity_service = EntityService()

        # Strategy request mock
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.strategy_name = GetSequenceSetsStrategy.__name__
        self.strategy_request.strategy_path = 'training_session'
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

        # Mock response data for HTTP requests
        self.mock_response_data = [
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

    @patch('training_session.strategy.TrainingSessionStrategy.requests.get')
    def test_apply_without_nested_requests(self, mock_get):
        # Setup mock response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response_data

        # Mock strategy executor to return a sequence set entity
        def mock_execute(session, request):
            sequence_set = SequenceSetEntity()
            session.add_child(sequence_set)
            request.ret_val = {'entity': sequence_set}
            return request
        self.strategy_executor.execute = mock_execute

        # Call the apply method
        self.strategy.apply(self.session_entity)

        # Verify that new nested requests were created
        self.assertEqual(len(self.strategy_request.get_nested_requests()), 2)

        # Get sequence sets using get_children_by_type
        sequence_sets = self.entity_service.get_children_ids_by_type(self.session_entity, EntityEnum.SEQUENCE_SET)
        sequence_sets = [self.entity_service.get_entity(uuid) for uuid in sequence_sets]

        # Validate the results
        self.assertEqual(len(sequence_sets), 2)

        first_sequence_set = sequence_sets[0]
        self.assertEqual(first_sequence_set.get_attribute('dataset_type'), self.dataset_type)
        self.assertEqual(first_sequence_set.get_attribute('sequence_length'), 10)
        self.assertEqual(first_sequence_set.get_attribute('start_timestamp'), '2023-01-01')
        self.assertEqual(first_sequence_set.get_attribute('metadata')['ticker'], 'AAPL')
        self.assertEqual(first_sequence_set.get_attribute('X_features'), self.X_features)
        self.assertEqual(first_sequence_set.get_attribute('y_features'), self.y_features)

        first_sequence = first_sequence_set.get_attribute('sequences')[0]
        self.assertEqual(first_sequence.sequence_length, 10)
        self.assertEqual(len(first_sequence.sequence_data), 10)
        self.assertTrue(first_sequence.start_timestamp > '2023-01-01')
        self.assertTrue(first_sequence.end_timestamp > first_sequence.start_timestamp)

    @patch('training_session.strategy.TrainingSessionStrategy.requests.get')
    def test_apply_with_existing_nested_requests(self, mock_get):
        # Setup mock response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.mock_response_data

        # Create pre-existing nested requests with specific UUIDs
        uuid1 = "test-uuid-1"
        uuid2 = "test-uuid-2"
        
        nested_request1 = StrategyRequestEntity()
        nested_request1.strategy_name = "CreateEntityStrategy"
        nested_request1.param_config = {
            'entity_class': "sequenceset_manager.entities.SequenceSetEntity.SequenceSetEntity",
            'entity_uuid': uuid1
        }
        
        nested_request2 = StrategyRequestEntity()
        nested_request2.strategy_name = "CreateEntityStrategy"
        nested_request2.param_config = {
            'entity_class': "sequenceset_manager.entities.SequenceSetEntity.SequenceSetEntity",
            'entity_uuid': uuid2
        }

        self.strategy_request.add_nested_request(nested_request1)
        self.strategy_request.add_nested_request(nested_request2)

        # Mock strategy executor to return sequence sets with specific UUIDs
        def mock_execute(session, request):
            sequence_set = SequenceSetEntity()
            sequence_set.uuid = request.param_config['entity_uuid']
            session.add_child(sequence_set)
            request.ret_val = {'entity': sequence_set}
            return request
        self.strategy_executor.execute = mock_execute

        # Call the apply method
        self.strategy.apply(self.session_entity)

        # Get sequence sets using get_children_by_type
        sequence_sets = self.entity_service.get_children_ids_by_type(self.session_entity, EntityEnum.SEQUENCE_SET)
        sequence_sets = [self.entity_service.get_entity(uuid) for uuid in sequence_sets]

        # Verify the number of sequence sets
        self.assertEqual(len(sequence_sets), 2)

        # Verify that the sequence sets have the correct UUIDs
        self.assertEqual(sequence_sets[0].uuid, uuid1)
        self.assertEqual(sequence_sets[1].uuid, uuid2)

        # Verify that no new nested requests were created
        self.assertEqual(len(self.strategy_request.get_nested_requests()), 2)

    def test_verify_executable_missing_params(self):
        # Test missing X_features
        invalid_request = StrategyRequestEntity()
        invalid_request.param_config = {
            'y_features': self.y_features,
            'model_set_configs': [],
            'dataset_type': self.dataset_type
        }
        with self.assertRaises(ValueError) as context:
            self.strategy.verify_executable(self.session_entity, invalid_request)
        self.assertTrue("Missing X_features in config" in str(context.exception))

        # Add more similar tests for other required parameters...
