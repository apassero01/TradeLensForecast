import numpy as np
import pandas as pd
from django.test import TestCase

from sequenceset_manager.models import SequenceSet
from sequenceset_manager.services import SequenceSetService
from training_manager.FeatureSetService import MeanVarianceScaler3D
from training_manager.TrainingSessionService import TrainingSessionService


class TrainingSessionServiceTestCase(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'open': [1, 2, 3, 4, 5], 'high': [2, 1.5, 2.5, 4, 5], 'close+1': [2, 3, 4, 5, 6]})
        # set dates to dataframe
        self.df.index = pd.date_range('2020-01-01', periods=5)

        self.sequence_set1 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 0, 'high': 1, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})

        sequences1 = SequenceSetService.create_sequence_objects(self.sequence_set1, self.df)
        self.sequence_set1.sequences = sorted(sequences1, key=lambda x: x.start_timestamp)


        self.sequence_set2 = SequenceSet.objects.create(dataset_type='stock', sequence_length=2,
                                                       start_timestamp='2020-01-01', end_timestamp='2020-01-05',
                                                       feature_dict={'open': 0, 'high': 1, 'close+1': 2},
                                                       metadata={'ticker': 'AAPL'})

        sequences2 = SequenceSetService.create_sequence_objects(self.sequence_set2, self.df)
        self.sequence_set2.sequences = sorted(sequences2, key=lambda x: x.start_timestamp)

        self.training_session_service = TrainingSessionService()

        for i, sequence in enumerate(self.sequence_set1.sequences + self.sequence_set2.sequences):
            sequence.id = i

    def test_create_training_session(self):
        X_features = ['open', 'high']
        y_features = ['close+1']
        sequence_params = {'sequence_length': 2}
        training_session = self.training_session_service.create_training_session(X_features, y_features, sequence_params)
        self.assertEqual(training_session.X_features, X_features)
        self.assertEqual(training_session.y_features, y_features)
        self.assertEqual(training_session.sequence_params, sequence_params)
        self.assertEqual(training_session.feature_dict, {'open': 0, 'high': 1, 'close+1': 2})
        self.assertEqual(training_session.X_feature_dict, {'open': 0, 'high': 1})
        self.assertEqual(training_session.y_feature_dict, {'close+1': 0})

    def test_create_3d_array(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        self.assertEqual(len(sequence_sets), 2)
        self.assertEqual(sequence_sets[0].X.shape, (4, 2, 2))
        self.assertEqual(sequence_sets[0].y.shape, (4, 1, 1))
        self.assertEqual(sequence_sets[1].X.shape, (4, 2, 2))
        self.assertEqual(sequence_sets[1].y.shape, (4, 1, 1))

    def test_train_test_split(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        self.training_session_service.train_test_split(training_session, '2020-01-04')
        self.assertEqual(training_session.sequence_sets[0].X_train.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[0].X_test.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[0].y_train.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[0].y_test.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[1].X_train.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[1].X_test.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[1].y_train.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[1].y_test.shape, (2, 1, 1))

        self.assertEqual(training_session.sequence_sets[0].train_seq_ids, [0, 1])
        self.assertEqual(training_session.sequence_sets[0].test_seq_ids, [2, 3])
        self.assertEqual(training_session.sequence_sets[1].train_seq_ids, [4, 5])
        self.assertEqual(training_session.sequence_sets[1].test_seq_ids, [6, 7])


    def test_combine_seq_sets(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        self.training_session_service.train_test_split(training_session, '2020-01-04')
        self.training_session_service.combine_seq_sets(training_session)
        self.assertEqual(training_session.X_train.shape, (4, 2, 2))
        self.assertEqual(training_session.X_test.shape, (4, 2, 2))
        self.assertEqual(training_session.y_train.shape, (4, 1, 1))
        self.assertEqual(training_session.y_test.shape, (4, 1, 1))
        np.testing.assert_almost_equal(training_session.train_seq_ids, np.array([0, 1, 4, 5]))
        np.testing.assert_almost_equal(training_session.test_seq_ids, np.array([2, 3, 6, 7]))

    def test_create_feature_sets(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        feature_set_configs = [
            {
            'type': 'X',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
            'feature_list': ['open', 'high']
            },
            {
            'type': 'y',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
            'feature_list': ['close+1']
            }
        ]
        self.training_session_service.create_feature_sets(training_session, feature_set_configs)
        self.assertEqual(len(training_session.sequence_sets[0].X_feature_sets), 1)
        self.assertEqual(len(training_session.sequence_sets[0].y_feature_sets), 1)
        self.assertEqual(len(training_session.sequence_sets[1].X_feature_sets), 1)
        self.assertEqual(len(training_session.sequence_sets[1].y_feature_sets), 1)


    def test_scale_sequences_X(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        feature_set_configs = [
            {
            'type': 'X',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
            'feature_list': ['open', 'high']
            },
            {
            'type': 'y',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
            'feature_list': ['close+1']
            }
        ]
        self.training_session_service.create_feature_sets(training_session, feature_set_configs)
        self.training_session_service.train_test_split(training_session, '2020-01-04')
        self.training_session_service.scale_sequence_sets_X(training_session)
        self.assertEqual(training_session.sequence_sets[0].X_train_scaled.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[0].X_test_scaled.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[1].X_train_scaled.shape, (2, 2, 2))
        self.assertEqual(training_session.sequence_sets[1].X_test_scaled.shape, (2, 2, 2))

        scaler = MeanVarianceScaler3D()
        X_train_scaled_actual = scaler.fit_transform(training_session.sequence_sets[0].X_train)
        X_test_scaled_actual = scaler.transform(training_session.sequence_sets[0].X_test)
        np.testing.assert_almost_equal(training_session.sequence_sets[0].X_train_scaled, X_train_scaled_actual)
        np.testing.assert_almost_equal(training_session.sequence_sets[0].X_test_scaled, X_test_scaled_actual)

        X_train_scaled_actual = scaler.fit_transform(training_session.sequence_sets[1].X_train)
        X_test_scaled_actual = scaler.transform(training_session.sequence_sets[1].X_test)
        np.testing.assert_almost_equal(training_session.sequence_sets[1].X_train_scaled, X_train_scaled_actual)
        np.testing.assert_almost_equal(training_session.sequence_sets[1].X_test_scaled, X_test_scaled_actual)


    def test_scale_sequences_y(self):
        training_session = self.training_session_service.create_training_session(['open', 'high'], ['close+1'], {'sequence_length': 2})
        training_session.sequence_sets = [self.sequence_set1, self.sequence_set2]
        sequence_sets = self.training_session_service.create_3d_array(training_session)
        feature_set_configs = [
            {
            'type': 'X',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                },
            'feature_list': ['open', 'high']
            },
            {
            'type': 'y',
                'scaler_config': {
                    'scaler_name': 'MEAN_VARIANCE_SCALER_3D'
                    },
            'feature_list': ['close+1']
            }
        ]
        self.training_session_service.create_feature_sets(training_session, feature_set_configs)
        self.training_session_service.train_test_split(training_session, '2020-01-04')
        self.training_session_service.scale_sequence_sets_y(training_session)
        self.assertEqual(training_session.sequence_sets[0].y_train_scaled.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[0].y_test_scaled.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[1].y_train_scaled.shape, (2, 1, 1))
        self.assertEqual(training_session.sequence_sets[1].y_test_scaled.shape, (2, 1, 1))

        scaler = MeanVarianceScaler3D()
        y_train_scaled_actual = scaler.fit_transform(training_session.sequence_sets[0].y_train)
        y_test_scaled_actual = scaler.transform(training_session.sequence_sets[0].y_test)
        np.testing.assert_almost_equal(training_session.sequence_sets[0].y_train_scaled, y_train_scaled_actual)
        np.testing.assert_almost_equal(training_session.sequence_sets[0].y_test_scaled, y_test_scaled_actual)

        y_train_scaled_actual = scaler.fit_transform(training_session.sequence_sets[1].y_train)
        y_test_scaled_actual = scaler.transform(training_session.sequence_sets[1].y_test)
        np.testing.assert_almost_equal(training_session.sequence_sets[1].y_train_scaled, y_train_scaled_actual)
        np.testing.assert_almost_equal(training_session.sequence_sets[1].y_test_scaled, y_test_scaled_actual)

    def test_create_feature_dict(self):
        X_features = ['open', 'high']
        y_features = ['close+1']
        training_session = self.training_session_service.create_training_session(X_features, y_features, {'sequence_length': 2})
        feature_dict = self.training_session_service.create_feature_dict(X_features, y_features)
        self.assertEqual(feature_dict, {'open': 0, 'high': 1, 'close+1': 2})

    def test_create_xy_feature_dict(self):
        X_features = ['open', 'high']
        y_features = ['close+1']
        training_session = self.training_session_service.create_training_session(X_features, y_features, {'sequence_length': 2})
        X_feature_dict, y_feature_dict = self.training_session_service.create_xy_feature_dict(X_features, y_features)
        self.assertEqual(X_feature_dict, {'open': 0, 'high': 1})
        self.assertEqual(y_feature_dict, {'close+1': 0})




