from django.test import TestCase
from peewee import Model

from training_session.models import ModelSet
from training_session.strategies import RetrieveSequenceSetsStrategy, CreateFeatureSetsStrategy


# Create your tests here.
class RetrieveSequenceSetsStrategyTestCase(TestCase):
    def setUp(self):
        self.X_features = ['open', 'close']
        self.y_features = ['close+1']
        self.dataset_type = 'stock'

        self.config = {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'training_session',
            'type': RetrieveSequenceSetsStrategy.__name__,
            'step_number': 1,
            'X_features': self.X_features,
            'y_features': self.y_features,
            'dataset_type': self.dataset_type,
            'model_set_configs': [
                {
                    'sequence_length': 10,
                    'start_timestamp': '2023-01-01',
                    'interval': '1d',
                    'ticker': 'AAPL',
                    'features': self.X_features + self.y_features,
                },
                {
                    'sequence_length': 20,
                    'start_timestamp': '2023-01-01',
                    'interval': '1d',
                    'ticker': 'AAPL',
                    'features': self.X_features + self.y_features,
                }
            ],
        }
        self.retrieve_sequence_sets_strategy = RetrieveSequenceSetsStrategy(self.config)

    def test_apply(self):
        model_sets = self.retrieve_sequence_sets_strategy.apply(model_sets = None)
        self.assertEqual(len(model_sets), 2)

        sequence_set = model_sets[0].data_set
        self.assertEqual(sequence_set.dataset_type, self.dataset_type)
        self.assertEqual(sequence_set.sequence_length, 10)
        self.assertEqual(sequence_set.start_timestamp, '2023-01-01')
        self.assertEqual(sequence_set.metadata['ticker'], 'AAPL')

        first_sequence = sequence_set.sequences[0]
        self.assertEqual(first_sequence.sequence_length, 10)
        self.assertEqual(len(first_sequence.sequence_data), 10)
        self.assertTrue(first_sequence.start_timestamp > '2023-01-01')
        self.assertTrue(first_sequence.end_timestamp > first_sequence.start_timestamp)


class CreateFeatureSetsStrategyTestCase(TestCase):
    def setUp(self):
        self.config = {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'training_session',
            'type': CreateFeatureSetsStrategy.__name__,
            'step_number': 1,
            'feature_set_configs': [
            {
                'feature_set_type': 'X',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
                'feature_list': ['open', 'high'],
                'do_fit_test': False
            },
            {
                'feature_set_type': 'y',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
                'feature_list': ['close+1'],
                'do_fit_test': False
            }
            ]
        }

        self.create_feature_sets_strategy = CreateFeatureSetsStrategy(self.config)
        self.model_sets = [ModelSet()]

    def test_apply(self):
        self.model_sets = self.create_feature_sets_strategy.apply(model_sets=self.model_sets)

        X_feature_sets = self.model_sets[0].X_feature_sets
        y_feature_sets = self.model_sets[0].y_feature_sets

        self.assertEqual(len(X_feature_sets), 1)
        self.assertEqual(len(y_feature_sets), 1)
        self.assertEqual(X_feature_sets[0].feature_list, ['open', 'high'])
        self.assertEqual(y_feature_sets[0].feature_list, ['close+1'])






