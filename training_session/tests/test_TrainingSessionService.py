from django.test import TestCase

from training_session.models import ModelSet
from training_session.services.TrainingSessionService import TrainingSessionService


class TrainingSessionServiceTestCase(TestCase):

    def setUp(self):
        self.session_service = TrainingSessionService()

    def test_create_training_session(self):
        session = self.session_service.create_training_session()
        self.assertEqual(session.status, 1)

    def test_initialize_params(self):
        session = self.session_service.create_training_session()
        session = self.session_service.initialize_params(session, ['feature1', 'feature2'], ['feature3'])
        self.assertEqual(session.X_features, ['feature1', 'feature2'])
        self.assertEqual(session.y_features, ['feature3'])

        self.assertDictEqual(session.X_feature_dict, {'feature1': 0, 'feature2': 1})
        self.assertDictEqual(session.y_feature_dict, {'feature3': 0})

    def test_populate_strategy_config(self):
        session = self.session_service.create_training_session()
        session.X_features = ['feature1', 'feature2']
        session.y_features = ['feature3']
        session.ordered_model_set_strategies = ['strategy1', 'strategy2']

        config = {
            'X_features': None,
            'y_features': None,
        }

        config = self.session_service.populate_strategy_config(session, config)
        self.assertDictEqual(config, {'X_features': ['feature1', 'feature2'], 'y_features': ['feature3']})


    def test_apply_model_set_strategy(self):
        session = self.session_service.create_training_session()
        session.X_features = ['feature1', 'feature2']
        session.y_features = ['feature3']

        model_set = ModelSet()
        model_set.X = [[1, 2], [3, 4]]
        model_set.y = [[5], [6]]

        session.model_sets = [model_set]

        config = {
            'm_service': 'training_session',
            'type': 'CreateFeatureSetsStrategy',
            'feature_set_configs': [
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
                    'feature_list': ['feature1', 'feature2'],
                    'do_fit_test': False,
                    'feature_set_type': 'X'
                },
                {
                    'scaler_config': {
                        'scaler_name': 'MEAN_VARIANCE_SCALER_3D',
                    },
                    'feature_list': ['feature3'],
                    'do_fit_test': False,
                    'feature_set_type': 'y'
                }
            ]
        }

        session = self.session_service.apply_model_set_strategy(session, config)

        self.assertEqual(len(session.model_sets), 1)
        self.assertEqual(len(session.ordered_model_set_strategies), 1)
        self.assertEqual(session.ordered_model_set_strategies[0]['step_number'], 0)

        self.assertEqual(len(session.model_sets[0].X_feature_sets), 1)
        self.assertEqual(len(session.model_sets[0].y_feature_sets), 1)

        self.assertEqual(session.model_sets[0].X_feature_sets[0].feature_list, ['feature1', 'feature2'])
        self.assertEqual(session.model_sets[0].y_feature_sets[0].feature_list, ['feature3'])
