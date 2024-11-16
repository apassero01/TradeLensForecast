from unittest.mock import patch, MagicMock

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

        session = self.session_service.initialize_params(session, ['feature1', 'feature2'], ['feature3'], ['model_set_config'], 'start_date')
        self.assertEqual(session.X_features, ['feature1', 'feature2'])
        self.assertEqual(session.y_features, ['feature3'])

        self.assertDictEqual(session.X_feature_dict, {'feature1': 0, 'feature2': 1})
        self.assertDictEqual(session.y_feature_dict, {'feature3': 0})


