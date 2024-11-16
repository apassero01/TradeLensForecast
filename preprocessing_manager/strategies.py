import numpy as np
import pandas as pd
from sympy import sequence

from shared_utils.strategies import ModelSetsStrategy


class PreprocessingStrategy(ModelSetsStrategy):
    def __init__(self, config):
        super().__init__(config)


class Create3dArraySequenceSetStrategy(PreprocessingStrategy):
    name = 'Create3dArr'
    def __init__(self, config):
        super().__init__(config)

    def apply(self, model_sets):
        for model_set  in model_sets:
            feature_dict = self.create_feature_dict(model_set.X_features, model_set.y_features)
            model_set.X, model_set.y, model_set.row_ids = self.create_3d_array_seq(model_set.data_set, model_set.X_features, model_set.y_features, feature_dict)
        return model_sets

    def create_feature_dict(self, X_features, y_features):
        feature_dict = {}
        for i, feature in enumerate(X_features + y_features):
            feature_dict[feature] = i

        return feature_dict

    def create_3d_array_seq(self, sequence_set, X_features, y_features, feature_dict):
        sequence_objs = sequence_set.sequences
        sequence_steps = len(sequence_objs[0].sequence_data)
        num_sequences = len(sequence_objs)
        # sequence set does not have a feature dict
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

        row_ids = np.array(row_ids)
        nan_mask = np.any(np.isnan(y), axis=(1, 2))

        X = X[~nan_mask]
        y = y[~nan_mask]
        row_ids = row_ids[~nan_mask]

        return X, y, row_ids.tolist()

    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': Create3dArraySequenceSetStrategy.__name__,
        }

class TrainTestSplitDateStrategy(PreprocessingStrategy):
    name = 'TrainTestSplitDate'
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
        split_date = pd.to_datetime(split_date).tz_localize(None)
        dates = [pd.to_datetime(date).tz_localize(None) for date in dates]

        if split_date not in dates:
            split_date = min(dates, key=lambda x: abs(x - split_date))

        if len(dates) != len(X):
            raise ValueError("Dates and X must be the same")

        split_index = dates.index(split_date)

        X_train, X_test = X[:split_index], X[split_index:]
        train_row_ids = model_set.row_ids[:split_index]
        test_row_ids = model_set.row_ids[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids

    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': TrainTestSplitDateStrategy.__name__,
            'split_date': None
        }


class ScaleByFeaturesStrategy(PreprocessingStrategy):
    name = 'ScaleByFeatures'
    def __init__(self, config):
        super().__init__(config)
        if config['m_service'] != 'preprocessing_manager':
            raise ValueError('m_service must be preprocessing_manager')
        if config['type'] != self.__class__.__name__:
            raise ValueError('type must be ScaleByFeaturesStrategy')
        if 'X_feature_dict' not in config:
            raise ValueError('X_feature_dict must be in config')
        if 'y_feature_dict' not in config:
            raise ValueError('y_feature_dict must be in config')

        self.required_keys += ['feature_set_type']


    def apply(self, model_sets):
        X_feature_dict = self.config['X_feature_dict']
        y_feature_dict = self.config['y_feature_dict']
        for model_set in model_sets:
            if model_set.X_feature_sets is not None and len(model_set.X_feature_sets) > 0:
                model_set.X_train_scaled, model_set.X_test_scaled = self.scale_X_by_features(model_set.X_feature_sets, model_set.X_train, model_set.X_test, X_feature_dict)
            if model_set.y_feature_sets is not None and len(model_set.y_feature_sets) > 0:
                model_set.y_train_scaled, model_set.y_test_scaled = self.scale_y_by_features(model_set.y_feature_sets, model_set.y_train, model_set.y_test, y_feature_dict)
            if model_set.Xy_feature_sets is not None and len(model_set.Xy_feature_sets) > 0:
                model_set.X_train_scaled, model_set.y_train_scaled = self.scale_Xy_by_features(model_set.Xy_feature_sets, model_set.X_train, model_set.y_train, X_feature_dict, y_feature_dict)
                model_set.X_test_scaled, model_set.y_test_scaled = self.scale_Xy_by_features(model_set.Xy_feature_sets, model_set.X_test, model_set.y_test, X_feature_dict, y_feature_dict)

        return model_sets

    def scale_X_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):
        arr1_scaled = np.zeros(arr1.shape)
        arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            arr1_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr1[:, :, arr1_feature_indices])

            if do_fit_test:
                arr2_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr2[:, :, arr1_feature_indices])
            else:
                arr2_scaled[:, :, arr1_feature_indices] = scaler.transform(arr2[:, :, arr1_feature_indices])

        return arr1_scaled, arr2_scaled

    def scale_y_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):
        '''
        Scale the y_feature sets. In practice, we only have a single y_feature_set so here we do not filter features by feature_dict
        but in the future, if for some reason we had y with different valued we would need to rework this.
        '''

        arr1_scaled = np.zeros(arr1.shape)
        arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            # Irrelivent for y features
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            arr1_scaled = scaler.fit_transform(arr1)

            if do_fit_test:
                arr2_scaled = scaler.fit_transform(arr2)
            else:
                arr2_scaled = scaler.transform(arr2)

        return arr1_scaled, arr2_scaled

    def scale_Xy_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict, arr2_feature_dict):
        #TODO BROKEN
        arr1_scaled = np.copy(arr1)
        arr2_scaled = np.copy(arr2)

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler

            arr1_feature_indices = [
                arr1_feature_dict[feature]
                for feature in feature_set.feature_list
                if feature in arr1_feature_dict
            ]
            arr2_feature_indices = [
                arr2_feature_dict[feature]
                for feature in feature_set.feature_list
                if feature in arr2_feature_dict
            ]

            # Extract features from arr1 and arr2
            arr1_features = arr1[:, :, arr1_feature_indices]  # Shape: (samples, time_steps, features)
            arr2_features = arr2[:, arr2_feature_indices, : ]  # Shape: (samples, time_steps, features)

            # Reshape arr1_features and arr2_features to (samples, time_steps * features)
            arr1_reshaped = arr1_features.reshape(arr1_features.shape[0], -1)
            arr2_reshaped = arr2_features.reshape(arr2_features.shape[0], -1)

            # Fit scaler on arr1_reshaped and transform arr2_reshaped
            arr1_scaled_flat = scaler.fit_transform(arr1_reshaped)
            arr2_scaled_flat = scaler.transform(arr2_reshaped)

            # Reshape back to original shapes
            arr1_scaled_features = arr1_scaled_flat.reshape(arr1_features.shape)
            arr2_scaled_features = arr2_scaled_flat.reshape(arr2_features.shape)

            arr1_scaled[:, :, arr1_feature_indices] = arr1_scaled_features
            arr2_scaled[:, arr2_feature_indices, :] = arr2_scaled_features

        return arr1_scaled, arr2_scaled


    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': ScaleByFeaturesStrategy.__name__,
            'X_feature_dict': None,
            'y_feature_dict': None,
        }


class CombineDataSetsStrategy(PreprocessingStrategy):
    '''
    Method to combine the data in multiple model sets into a combined data source
    requires each model_set to have X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, train_row_ids, test_row_ids
    Note: if future needs require data to be extracted from different arr names ie X_train then another strategy will be needed or we will have to abstract better
    '''
    name = 'CombineDataSets'
    def __init__(self, config):
        super().__init__(config)
        config['is_final'] = True

        self.required_keys += ['is_final']

    def apply(self, model_sets):
        X_train = np.concatenate([model_set.X_train_scaled for model_set in model_sets])
        X_test = np.concatenate([model_set.X_test_scaled for model_set in model_sets])
        y_train = np.concatenate([model_set.y_train_scaled for model_set in model_sets])
        y_test = np.concatenate([model_set.y_test_scaled for model_set in model_sets])
        train_row_ids = np.concatenate([model_set.train_row_ids for model_set in model_sets])
        test_row_ids = np.concatenate([model_set.test_row_ids for model_set in model_sets])


        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids


    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'preprocessing_manager',
            'type': CombineDataSetsStrategy.__name__,
            'is_final': True
        }






