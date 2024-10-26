from django.test import TestCase

from shared_utils.strategies import ModelSetsStrategy

# Create your tests here.
class ModelSetsStrategyTestCase(TestCase):
    def test_get_strategy_instance(self):
        config = {
            'step_number': 1,
            'm_service': 'training_session',
            'type': 'RetrieveSequenceSetsStrategy',
            'X_features': ['feature1', 'feature2'],
            'y_features': ['feature3'],
            'sequence_params': [{'sequence_length': 5, 'start_timestamp': '2020-01-01'}],
            'dataset_type': 'stock'
        }
        strategy = ModelSetsStrategy.get_strategy_instance(config)

        self.assertEqual(strategy.get_step_number(), 1)
        self.assertEqual(strategy.get_m_service(), 'training_session')
        self.assertEqual(strategy.get_type(), 'RetrieveSequenceSetsStrategy')
        self.assertEqual(strategy.get_config(), config)
        self.assertEqual(strategy.config['is_applied'], False)

