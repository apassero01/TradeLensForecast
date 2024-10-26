import numpy as np
import pandas as pd


class PreprocessingService:
    def create_3d_array(self, sequence_objs, X_features, y_features, feature_dict):

        sequence_steps = len(sequence_objs[0].sequence_data)
        num_sequences = len(sequence_objs)

        X = np.zeros((num_sequences, sequence_steps, len(X_features)))
        y = np.zeros((num_sequences, len(y_features), 1))

        sequence_ids = []
        for i, sequence_obj in enumerate(sequence_objs):
            X_cols = [feature_dict[col] for col in X_features]
            y_cols = [feature_dict[col] for col in y_features]

            X[i, :, :] = np.array(sequence_obj.sequence_data)[:, X_cols]

            temp_y = np.array(sequence_obj.sequence_data)[-1, y_cols]
            y[i, :, :] = temp_y.reshape(-1, 1)
            sequence_ids.append(sequence_obj.id)

        return X, y, sequence_ids

    def combine_seq_sets(self, sequence_sets):
        X_train = np.concatenate([sequence_set.X_train for sequence_set in sequence_sets])
        X_test = np.concatenate([sequence_set.X_test for sequence_set in sequence_sets])
        y_train = np.concatenate([sequence_set.y_train for sequence_set in sequence_sets])
        y_test = np.concatenate([sequence_set.y_test for sequence_set in sequence_sets])
        train_seq_ids = np.concatenate([sequence_set.train_seq_ids for sequence_set in sequence_sets])
        test_seq_ids = np.concatenate([sequence_set.test_seq_ids for sequence_set in sequence_sets])

        return X_train, X_test, y_train, y_test, train_seq_ids, test_seq_ids


    # def create_3d_array_seqsets(self, sequence_sets, future_predictions=False):
    #     for sequence_set in sequence_sets:
    #             X, y = self.create_3d_array(sequence_set.training_session, sequence_set.sequences)
    #             if not future_predictions:
    #                 for i in range(len(X) - 1, 0, -1):
    #                     if np.isnan(y[i]).any():
    #                         X = np.delete(X, i, axis=0)
    #                         y = np.delete(y, i, axis=0)
    #
    #             sequence_set.X = X
    #             sequence_set.y = y

    def scale(self, scaler, arr1, arr2 = None):
        '''
        Scale the data
        '''
        arr1_scaled = scaler.fit_transform(arr1)
        if arr2 is not None:
            arr2_scaled = scaler.transform(arr2)

        if arr2 is not None:
            return arr1_scaled, arr2_scaled
        else:
            return arr1_scaled

    def scale_by_features(self, feature_sets, feature_dict, arr1, arr2 = None):
        '''
        Scale the data
        '''
        arr1_scaled = np.zeros(arr1.shape)
        if arr2 is not None:
            arr2_scaled = np.zeros(arr2.shape)

        for feature_set in feature_sets:
            scaler = feature_set.scaler
            feature_indices = [feature_dict[feature] for feature in feature_set.X_feature_list]
            arr1_scaled[:, :, feature_indices] = scaler.fit_transform(arr1[:, :, feature_indices])
            if arr2 is not None:
                arr2_scaled[:, :, feature_indices] = scaler.transform(arr2[:, :, feature_indices])

        if arr2 is not None:
            return arr1_scaled, arr2_scaled
        else:
            return arr1_scaled


    def train_test_split(self, X, y, dates, split_date, sequence_ids):
        '''
        Split the data into training and testing sets
        '''
        if len(dates) != len(X):
            raise ValueError("Dates and X must be the same")
        # dates is a 1d list
        if split_date not in dates:
            split_date = min(dates, key=lambda x: abs(pd.to_datetime(x) - pd.to_datetime(split_date)))
        split_index = dates.index(split_date)
        X_train, X_test = X[:split_index], X[split_index:]
        train_seq_ids = sequence_ids[:split_index]
        test_seq_ids = sequence_ids[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test, train_seq_ids, test_seq_ids