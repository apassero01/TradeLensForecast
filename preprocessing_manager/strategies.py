import numpy as np
import pandas as pd
from sympy import sequence

from shared_utils.strategies import ModelSetsStrategy


class PreprocessingStrategy(ModelSetsStrategy):
    def __init__(self, config):
        super().__init__(config)


class Create3dArraySequenceSetStrategy(PreprocessingStrategy):
    def __init__(self, config):
        super().__init__(config)

    def apply(self, model_sets):
        for model_set  in model_sets:
            model_set.X, model_set.y, model_set.row_ids = self.create_3d_array_seq(model_set.data_set, model_set.X_features, model_set.y_features)
        return model_sets

    def create_3d_array_seq(self, sequence_set, X_features, y_features):
        sequence_objs = sequence_set.sequences
        sequence_steps = len(sequence_objs[0].sequence_data)
        num_sequences = len(sequence_objs)
        feature_dict = sequence_set.feature_dict

        X = np.zeros((num_sequences, sequence_steps, len(X_features)))
        y = np.zeros((num_sequences, len(y_features), 1))

        row_ids = []
        for i, sequence_obj in enumerate(sequence_objs):
            X_cols = [feature_dict[col] for col in X_features]
            y_cols = [feature_dict[col] for col in y_features]

            X[i, :, :] = np.array(sequence_obj.sequence_data)[:, X_cols]

            temp_y = np.array(sequence_obj.sequence_data)[-1, y_cols]
            y[i, :, :] = temp_y.reshape(-1, 1)
            row_ids.append(sequence_obj.id)

        return X, y, row_ids

class TrainTestSplitDateStrategy(PreprocessingStrategy):
    def __init__(self, config):
        super().__init__(config)

        if 'split_date' not in config:
            raise ValueError('split_date must be in config')

        self.required_keys += ['split_date']

    def apply(self, model_sets):
        for model_set in model_sets:
            model_set.X_train, model_set.X_test, model_set.y_train, model_set.y_test, model_set.train_row_ids, model_set.test_row_ids = self.train_test_split(model_set, self.config['split_date'])
        return model_sets

    def train_test_split(self, model_set, split_date):
        X = model_set.X
        y = model_set.y

        sequence_set = model_set.data_set
        dates = [sequence.end_timestamp for sequence in sequence_set.sequences]

        if len(dates) != len(X):
            raise ValueError("Dates and X must be the same")

        if split_date not in dates:
            split_date = min(dates, key=lambda x: abs(pd.to_datetime(x) - pd.to_datetime(split_date)))
        split_index = dates.index(split_date)

        X_train, X_test = X[:split_index], X[split_index:]
        train_row_ids = model_set.row_ids[:split_index]
        test_row_ids = model_set.row_ids[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids


class ScaleByFeaturesStrategy(PreprocessingStrategy):
    def __init__(self, config):
        super().__init__(config)
        if config['m_service'] != 'preprocessing_manager':
            raise ValueError('m_service must be preprocessing_manager')
        if config['type'] != self.__class__.__name__:
            raise ValueError('type must be ScaleByFeaturesStrategy')
        if 'feature_set_type' not in config:
            raise ValueError('feature_set_type must be in config')
        if config['feature_set_type'] not in ['X', 'y', 'Xy']:
            raise ValueError('feature_set_type must be X or y or Xy')

        self.required_keys += ['feature_set_type']


    def apply(self, model_sets):
        for model_set in model_sets:
            if self.config['feature_set_type'] == 'X':
                model_set.X_train_scaled, model_set.X_test_scaled = self.scale_by_features(model_set.X_feature_sets, model_set.X_train, model_set.X_test, model_set.X_feature_dict)
            elif self.config['feature_set_type'] == 'y':
                model_set.y_train_scaled, model_set.y_test_scaled = self.scale_by_features(model_set.y_feature_sets, model_set.y_train, model_set.y_test, model_set.y_feature_dict)
            elif self.config['feature_set_type'] == 'Xy':
                model_set.X_train_scaled, model_set.y_train_scaled = self.scale_by_features(model_set.Xy_feature_sets, model_set.X_train, model_set.y_train, model_set.X_feature_dict, model_set.y_feature_dict)
                model_set.X_test_scaled, model_set.y_test_scaled = self.scale_by_features(model_set.Xy_feature_sets, model_set.X_test, model_set.y_test, model_set.X_feature_dict, model_set.y_feature_dict)

        return model_sets


    def scale_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict, arr2_feature_dict = None):
        '''
        Scale the data
        '''

        arr1_scaled = np.zeros(arr1.shape)
        arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            arr1_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr1[:, :, arr1_feature_indices])

            if arr2_feature_dict is None:
                if do_fit_test:
                    arr2_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr2[:, :, arr1_feature_indices])
                else:
                    arr2_scaled[:, :, arr1_feature_indices] = scaler.transform(arr2[:, :, arr1_feature_indices])
            else:
                # Special case: arr1 and arr2 differ, so flatten each feature to a 1D array
                # When arr2 is scaled similarly to arr1 (standard case), apply the transformation directly
                arr1_flat = arr1[:, :, arr1_feature_indices].reshape(-1, len(arr1_feature_indices))
                arr1_scaled_flat = scaler.fit_transform(arr1_flat)
                arr1_scaled[:, :, arr1_feature_indices] = arr1_scaled_flat.reshape(arr1.shape[0], arr1.shape[1], -1)

                arr2_indices = [arr2_feature_dict[feature] for feature in feature_set.secondary_feature_list]
                arr2_flat = arr2[:, :, arr2_indices].reshape(-1, len(arr2_indices))
                arr2_scaled_flat = scaler.transform(arr2_flat)
                arr2_scaled[:, :, arr2_indices] = arr2_scaled_flat.reshape(arr2.shape[0], arr2.shape[1], -1)

        return arr1_scaled, arr2_scaled


class CombineDataSetsStrategy(PreprocessingStrategy):
    def __init__(self, config):
        super().__init__(config)
        config['is_final'] = True

        self.required_keys += ['is_final']

    def apply(self, model_sets):
        X_train = np.concatenate([model_set.X_train for model_set in model_sets])
        X_test = np.concatenate([model_set.X_test for model_set in model_sets])
        y_train = np.concatenate([model_set.y_train for model_set in model_sets])
        y_test = np.concatenate([model_set.y_test for model_set in model_sets])
        train_row_ids = np.concatenate([model_set.train_row_ids for model_set in model_sets])
        test_row_ids = np.concatenate([model_set.test_row_ids for model_set in model_sets])

        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids






