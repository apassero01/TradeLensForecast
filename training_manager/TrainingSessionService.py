from abc import ABC

import numpy as np

from dataset_manager.services import FeatureFactoryService
from training_manager.FeatureSetService import FeatureSetService
from training_manager.PreprocessingService import PreprocessingService
from training_manager.models import FeatureSet, TrainingSession


class TrainingSessionService(ABC):

    def __init__(self):
        self.preprocessing_service = PreprocessingService()
        self.feature_set_service = FeatureSetService()

    def create_training_session(self, X_features, y_features, sequence_params):
        training_session = TrainingSession.objects.create(X_features=X_features,
                                                           y_features=y_features,
                                                           sequence_params=sequence_params)
        training_session.feature_dict = self.create_feature_dict(X_features, y_features)
        training_session.X_feature_dict, training_session.y_feature_dict = self.create_xy_feature_dict(X_features, y_features)

        return training_session

    def retrieve_sequence_sets(self, training_session):
        pass

    def create_3d_array(self, training_session):
        sequence_sets = training_session.sequence_sets

        for sequence_set in sequence_sets:
            sequence_set.X, sequence_set.y, sequence_set.sequence_ids = self.preprocessing_service.create_3d_array(sequence_set.sequences, training_session.X_features, training_session.y_features, training_session.feature_dict)

        return sequence_sets

    def create_feature_sets(self, training_session, feature_set_configs):
        for sequence_set in training_session.sequence_sets:
            sequence_set.X_feature_sets = []
            sequence_set.y_feature_sets = []
            for config in feature_set_configs:
                feature_set = self.feature_set_service.create_feature_set(config)
                if config['type'] == 'X':
                    sequence_set.X_feature_sets.append(feature_set)
                else:
                    sequence_set.y_feature_sets.append(feature_set)

    def train_test_split(self, training_session, split_date ):
        for sequence_set in training_session.sequence_sets:
            dates = [sequence.end_timestamp for sequence in sequence_set.sequences]
            sequence_set.X_train, sequence_set.X_test, sequence_set.y_train, sequence_set.y_test, sequence_set.train_seq_ids, sequence_set.test_seq_ids = self.preprocessing_service.train_test_split(X = sequence_set.X, y = sequence_set.y,
                                                                                                                                                                                                      dates = dates, split_date = split_date,
                                                                                                                                                                                                      sequence_ids = sequence_set.sequence_ids)
    def combine_seq_sets(self, training_session):
        training_session.X_train, training_session.X_test, training_session.y_train, training_session.y_test, training_session.train_seq_ids, training_session.test_seq_ids = self.preprocessing_service.combine_seq_sets(training_session.sequence_sets)


    def scale_sequence_sets_X(self, training_session):
        for sequence_set in training_session.sequence_sets:
            sequence_set.X_train_scaled, sequence_set.X_test_scaled = self.preprocessing_service.scale_by_features(sequence_set.X_feature_sets, training_session.X_feature_dict, sequence_set.X_train, sequence_set.X_test)

    def scale_sequence_sets_y(self, training_session):
        for sequence_set in training_session.sequence_sets:
            sequence_set.y_train_scaled, sequence_set.y_test_scaled = self.preprocessing_service.scale_by_features(sequence_set.y_feature_sets, training_session.y_feature_dict, sequence_set.y_train, sequence_set.y_test)


    def create_feature_dict(self, X_features, y_features):
        indices_seq = np.arange(len(X_features) + len(y_features))
        feature_dict = {col : index for col, index in zip(X_features + y_features, indices_seq)}
        return feature_dict

    def create_xy_feature_dict(self, X_features, y_features):
        X_indices_seq = np.arange(len(X_features))
        y_indices_seq = np.arange(len(y_features))

        X_feature_dict = {col : index for col, index in zip(X_features, X_indices_seq)}
        y_feature_dict = {col : index for col, index in zip(y_features, y_indices_seq)}

        return X_feature_dict, y_feature_dict