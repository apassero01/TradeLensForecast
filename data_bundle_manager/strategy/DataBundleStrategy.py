from copy import deepcopy

import numpy as np
import pandas as pd

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.entities.services.FeatureSetEntityService import FeatureSetEntityService
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy, GetEntityStrategy


class DataBundleStrategy(Strategy):
    entity_type = EntityEnum.DATA_BUNDLE
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        NotImplementedError("Child classes must implement the 'apply' method.")

    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}


# class LoadDataBundleDataStrategy(DataBundleStrategy):
#     name = "LoadDataBundleData"

#     def __init__(self, strategy_executor, strategy_request):
#         super().__init__(strategy_executor, strategy_request)
#         self.strategy_request = strategy_request
#         self.strategy_executor = strategy_executor

#     def apply(self, data_bundle):
#         param_config = self.strategy_request.param_config
#         data_bundle.set_dataset(param_config)

#         return self.strategy_request

#     def verify_executable(self, entity, strategy_request):
#         raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

#     @staticmethod
#     def get_request_config():
#         return {
#             'strategy_name': LoadDataBundleDataStrategy.__name__,
#             'strategy_path': None,
#             'param_config': {}
#         }

class CreateFeatureSetsStrategy(DataBundleStrategy):
    name = "CreateFeatureSets"
    strategy_description = 'Creates and configures multiple FeatureSetEntity children for data preprocessing pipelines. Takes feature_set_configs specifying scaler types, feature lists, and scaling behaviors, generates child entities via CreateEntityStrategy, configures each with scaler objects and feature mappings, assigns X_feature_dict and y_feature_dict from parent bundle, and saves all entities. Manages nested strategy execution and ensures proper parent-child relationships for downstream data processing workflows.'

    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)
        self.feature_set_entity_service = FeatureSetEntityService()

    def apply(self, data_bundle):
        param_config = self.strategy_request.param_config
        feature_sets = self.get_child_feature_sets(len(param_config['feature_set_configs']), data_bundle)
        if len(feature_sets) != len(param_config['feature_set_configs']):
            raise ValueError("Number of feature sets created does not match number of feature set configs")

        for feature_set, config in zip(feature_sets, param_config['feature_set_configs']):
            feature_set = self.feature_set_entity_service.create_feature_set(config, feature_set)
            feature_set.set_attribute('X_feature_dict', data_bundle.get_attribute('X_feature_dict'))
            feature_set.set_attribute('y_feature_dict', data_bundle.get_attribute('y_feature_dict'))
            self.entity_service.save_entity(feature_set)

        return self.strategy_request

    def get_child_feature_sets(self, num_feature_sets, data_bundle):
        feature_sets = []
        nested_requests = deepcopy(self.strategy_request.get_nested_requests())

        if len(nested_requests) != num_feature_sets:
            nested_requests = []
            for i in range(num_feature_sets):
                strat_request = StrategyRequestEntity()
                strat_request.strategy_name = CreateEntityStrategy.__name__
                strat_request.target_entity_id = data_bundle.entity_id
                strat_request.param_config['entity_class'] =  'data_bundle_manager.entities.FeatureSetEntity.FeatureSetEntity'
                nested_requests.append(strat_request)

        for strat_request in nested_requests:
            feature_set = self.executor_service.execute(data_bundle, strat_request).ret_val['child_entity']
            feature_sets.append(feature_set)
            self.strategy_request.add_nested_request(strat_request)

        return feature_sets


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if 'feature_set_configs' not in param_config.keys():
            raise ValueError("Missing feature_set_configs in config")
        for feature_set_config in param_config['feature_set_configs']:
            if 'feature_set_type' not in feature_set_config.keys():
                raise ValueError("Missing feature_set_type in feature_set_config")
            if 'feature_list' not in feature_set_config.keys():
                raise ValueError("Missing feature_list in feature_set_config")
            if 'scaler_config' not in feature_set_config.keys():
                raise ValueError("Missing scaler_config in feature_set_config")
            if 'do_fit_test' not in feature_set_config.keys():
                raise ValueError("Missing do_fit_test in feature_set_config")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CreateFeatureSetsStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'feature_set_configs': [
                    {
                        'scaler_config': {
                            'scaler_name': 'MIN_MAX_SEQ_BY_SEQ_2D'
                        },
                        'feature_list': [],
                        'feature_set_type': 'X',
                        'do_fit_test': False,
                        'secondary_feature_list': None

                    }
                ]
            }
        }


class SplitBundleDateStrategy(DataBundleStrategy):
    name = "SplitBundleDate"
    strategy_description = 'Performs temporal data splitting based on a specified date threshold for time series data. Reads X, y, row_ids, and seq_end_dates from data bundle, converts split_date to timezone-naive datetime, iterates through sequence end dates to classify data as training (before split_date) or testing (after split_date), creates separate numpy arrays for X_train, X_test, y_train, y_test, and corresponding row ID lists, then stores all split datasets back in the data bundle for downstream model training and evaluation.'
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        self.verify_executable(data_bundle, self.strategy_request)
        param_config = self.strategy_request.param_config
        X_train, X_test, y_train, y_test, train_row_ids, test_row_ids = self.train_test_split(data_bundle, param_config['split_date'])
        data_bundle.set_attribute('X_train', X_train)
        data_bundle.set_attribute('X_test', X_test)
        data_bundle.set_attribute('y_train', y_train)
        data_bundle.set_attribute('y_test', y_test)
        data_bundle.set_attribute('train_row_ids', train_row_ids)
        data_bundle.set_attribute('test_row_ids', test_row_ids)

        self.strategy_request.ret_val['entity'] = data_bundle

        return self.strategy_request

    def train_test_split(self, data_bundle, split_date):
        X = data_bundle.get_attribute('X')
        y = data_bundle.get_attribute('y')
        row_ids = data_bundle.get_attribute('row_ids')

        split_date = pd.to_datetime(split_date).tz_localize(None)
        date_list = data_bundle.get_attribute('seq_end_dates')
        date_list = [pd.to_datetime(date).tz_localize(None) for date in date_list]

        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []
        train_row_ids, test_row_ids = [], []

        for i, date in enumerate(date_list):
            if date > split_date:
                X_test_list.append(X[i])
                y_test_list.append(y[i])
                test_row_ids.append(row_ids[i])
            else:
                X_train_list.append(X[i])
                y_train_list.append(y[i])
                train_row_ids.append(row_ids[i])

        # Convert lists to numpy arrays
        X_train = np.array(X_train_list)
        X_test = np.array(X_test_list)
        y_train = np.array(y_train_list)
        y_test = np.array(y_test_list)

        return X_train, X_test, y_train, y_test, train_row_ids, test_row_ids

    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if 'split_date' not in param_config.keys():
            raise ValueError("Missing split_date in config")
        if not entity.has_attribute('X'):
            raise ValueError("Missing X in dataset")
        if not entity.has_attribute('y'):
            raise ValueError("Missing y in dataset")
        if not entity.has_attribute('row_ids'):
            raise ValueError("Missing row_ids in dataset")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': SplitBundleDateStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                'split_date': None,
            }
        }


class ScaleByFeatureSetsStrategy(DataBundleStrategy):
    name = "ScaleByFeatureSets"
    strategy_description = 'Applies feature-wise scaling transformations to training and testing datasets using configured FeatureSetEntity scalers. Retrieves child feature sets and categorizes them by type (X, y, or Xy), initializes scaled array containers, applies scaler.fit_transform() on training data and scaler.transform() on test data (with optional fit_test behavior), handles multi-dimensional feature indexing through feature dictionaries, and stores X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled back in the data bundle. Supports complex Xy feature sets that combine input and output features for joint scaling operations.'
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundle):
        self.verify_executable(data_bundle, self.strategy_request)
        param_config = self.strategy_request.param_config
        X_feature_dict = data_bundle.get_attribute('X_feature_dict')
        y_feature_dict = data_bundle.get_attribute('y_feature_dict')
        feature_set_ids = self.entity_service.get_children_ids_by_type(data_bundle, EntityEnum.FEATURE_SET)
        feature_sets = [self.get_feature_set(feature_set_id) for feature_set_id in feature_set_ids]



        X_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'X']
        y_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'y']
        Xy_feature_sets = [feature_set for feature_set in feature_sets if feature_set.feature_set_type == 'Xy']

        self.X_train_scaled = np.zeros (data_bundle.get_attribute('X_train').shape)
        self.X_test_scaled = np.zeros (data_bundle.get_attribute('X_test').shape)
        self.y_test_scaled = np.zeros (data_bundle.get_attribute('y_test').shape)
        self.y_train_scaled = np.zeros (data_bundle.get_attribute('y_train').shape)


        if len(X_feature_sets) > 0:
            self.scale_X_by_features(X_feature_sets, data_bundle.get_attribute('X_train'), data_bundle.get_attribute('X_test'), X_feature_dict)
        if len(y_feature_sets) > 0:
            self.scale_y_by_features(y_feature_sets, data_bundle.get_attribute('y_train'), data_bundle.get_attribute('y_test'), y_feature_dict)
        if len(Xy_feature_sets) > 0:
            self.scale_Xy_by_features(Xy_feature_sets, data_bundle.get_attribute('X_train'), data_bundle.get_attribute('y_train'), X_feature_dict, y_feature_dict, self.X_train_scaled, self.y_train_scaled)
            self.scale_Xy_by_features(Xy_feature_sets, data_bundle.get_attribute('X_test'), data_bundle.get_attribute('y_test'), X_feature_dict, y_feature_dict, self.X_test_scaled, self.y_test_scaled)

        data_bundle.set_attribute('X_train_scaled', self.X_train_scaled)
        data_bundle.set_attribute('X_test_scaled', self.X_test_scaled)
        data_bundle.set_attribute('y_train_scaled', self.y_train_scaled)
        data_bundle.set_attribute('y_test_scaled', self.y_test_scaled)

        for feature_set in feature_sets:
            self.entity_service.save_entity(feature_set)

        self.strategy_request.ret_val['entity'] = data_bundle
        return self.strategy_request

    def scale_X_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            self.X_train_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr1[:, :, arr1_feature_indices])

            if do_fit_test:
                self.X_test_scaled[:, :, arr1_feature_indices] = scaler.fit_transform(arr2[:, :, arr1_feature_indices])
            else:
                self.X_test_scaled[:, :, arr1_feature_indices] = scaler.transform(arr2[:, :, arr1_feature_indices])

    def scale_y_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict):
        '''
        Scale the y_feature sets. In practice, we only have a single y_feature_set so here we do not filter features by feature_dict
        but in the future, if for some reason we had y with different valued we would need to rework this.
        '''

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler
            # Irrelivent for y features
            arr1_feature_indices = [arr1_feature_dict[feature] for feature in feature_set.feature_list]
            self.X_train_scaled = scaler.fit_transform(arr1)

            if do_fit_test:
                self.X_test_scaled = scaler.fit_transform(arr2)
            else:
                self.X_test_scaled = scaler.transform(arr2)

    def scale_Xy_by_features(self, feature_sets, arr1, arr2, arr1_feature_dict, arr2_feature_dict, target_arr1, target_arr2):

        for feature_set in feature_sets:
            do_fit_test = feature_set.do_fit_test
            scaler = feature_set.scaler

            # Extract feature indices for arr1 and arr2
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
            arr2_features = arr2[:, arr2_feature_indices, :]  # Shape: (samples, features, time_steps)

            # Flatten arr1 and arr2 to 2D arrays
            arr1_reshaped = arr1_features.reshape(arr1_features.shape[0], -1)
            arr2_reshaped = arr2_features.reshape(arr2_features.shape[0], -1)

            # Fit and transform arr1, and transform arr2
            arr1_scaled_flat = scaler.fit_transform(arr1_reshaped)
            arr2_scaled_flat = scaler.transform(arr2_reshaped)

            # Reshape scaled data back to original 3D shapes
            arr1_scaled_features = arr1_scaled_flat.reshape(arr1_features.shape)
            arr2_scaled_features = arr2_scaled_flat.reshape(arr2_features.shape)

            # Update scaled arrays
            target_arr1[:, :, arr1_feature_indices] = arr1_scaled_features
            target_arr2[:, arr2_feature_indices, :] = arr2_scaled_features


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config
        if not entity.has_attribute('X_feature_dict'):
            raise ValueError("Missing X_feature_dict in dataset")
        if not entity.has_attribute('y_feature_dict'):
            raise ValueError("Missing y_feature_dict in dataset")
        if not entity.has_attribute('X_train'):
            raise ValueError("Missing X in dataset")
        if not entity.has_attribute('y_train'):
            raise ValueError("Missing y in dataset")
        if not entity.has_attribute('X_test'):
            raise ValueError("Missing X_test in dataset")
        if not entity.has_attribute('y_test'):
            raise ValueError("Missing y_test in dataset")

    def get_feature_set(self, entity_id):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = GetEntityStrategy.__name__
        strategy_request.target_entity_id = entity_id

        strategy_request = self.executor_service.execute_request(strategy_request)
        self.strategy_request.add_nested_request(strategy_request)

        return strategy_request.ret_val['entity']

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': ScaleByFeatureSetsStrategy.__name__,
            'strategy_path': None,
            'param_config': {}
        }

class InverseScaleByFeatureSetsStrategy(Strategy):
    name = "InverseScaleByFeatureSets"
    strategy_description = 'Reverses scaling transformations on predicted or processed data to restore original value ranges. Takes a target_array_attribute from entity containing scaled data, retrieves feature sets from entity children, categorizes them into y and Xy types, applies inverse_transform() using stored scalers with proper feature indexing via y_feature_dict, handles both simple y feature sets and complex Xy feature sets requiring 2D reshaping, and stores the inverse-transformed array in a configurable output_attribute. Essential for converting model predictions back to interpretable real-world scales.'

    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, entity):
        # Verify that the entity has the required target attribute.
        self.verify_executable(entity, self.strategy_request)
        param_config = self.strategy_request.param_config

        # Get the name of the attribute that holds the array to inverse transform.
        target_array_attribute = param_config.get("target_array_attribute")
        # Optionally allow an output attribute name; otherwise default to "inversed_{target_array_attribute}".
        output_attribute = param_config.get("output_attribute", f"inversed_{target_array_attribute}")

        # Retrieve the array from the entity (assumed to be shape (batch, seq_length, 1)).
        arr = entity.get_attribute(target_array_attribute)

        # Retrieve feature sets from the children of the parent entity.
        feature_set_ids = self.entity_service.get_children_ids_by_type(entity, EntityEnum.FEATURE_SET)
        feature_sets = [self.get_feature_set(fs_id) for fs_id in feature_set_ids]

        if not feature_sets:
            raise ValueError("No feature sets found.")

        # Assume the y feature dictionary is stored in the first feature set.
        y_feature_dict = feature_sets[0].get_attribute('y_feature_dict')

        # Only consider y and Xy feature sets.
        y_feature_sets = [fs for fs in feature_sets if fs.feature_set_type == 'y']
        Xy_feature_sets = [fs for fs in feature_sets if fs.feature_set_type == 'Xy']

        # Make a working copy of the array to apply inverse transformation.
        inverse_transformed_array = np.copy(arr)

        # Inverse transform for y feature sets.
        if y_feature_sets:
            self.inverse_scale_y_by_features(y_feature_sets, inverse_transformed_array, y_feature_dict)

        # Inverse transform for Xy feature sets.
        if Xy_feature_sets:
            self.inverse_scale_Xy_by_features(Xy_feature_sets, inverse_transformed_array, y_feature_dict)

        # Save the new, inverse-transformed array into the entity.
        entity.set_attribute(output_attribute, inverse_transformed_array)

        # Optionally, persist any changes to the feature set entities.
        for fs in feature_sets:
            self.entity_service.save_entity(fs)

        self.strategy_request.ret_val['entity'] = entity

        return self.strategy_request

    def inverse_scale_y_by_features(self, feature_sets, arr, feature_dict):
        """
        For each y feature set, extract the slice of the y array corresponding to the features (via the feature dict)
        and perform the inverse transformation using the stored scaler.
        Now the feature dimension is assumed to be along the second axis (i.e. arr has shape (batch, features, seq_length)).
        """
        for fs in feature_sets:
            scaler = fs.scaler
            # Get the feature indices from the y feature dictionary.
            feature_indices = [feature_dict[feature] for feature in fs.feature_list if feature in feature_dict]
            if not feature_indices:
                continue
            # Process along the second dimension (features).
            slice_ = arr[:, feature_indices, :]  # Shape: (batch, num_features, seq_length)
            # Apply inverse transformation (reshape if necessary depending on scaler's requirements).
            inv_slice = scaler.inverse_transform(slice_)
            # Replace the original slice with the inverse-transformed data.
            arr[:, feature_indices, :] = inv_slice

    def inverse_scale_Xy_by_features(self, feature_sets, arr, feature_dict):
        """
        For each Xy feature set, extract the corresponding slice from the y array and apply inverse transformation.
        Because the forward transformation may have reshaped the data, we reshape to 2D, inverse transform, then reshape back.
        Now we assume the feature dimension is along the second axis.
        """
        fs = feature_sets[0]  # Assume only one Xy feature set.
        # for fs in feature_sets:
        scaler = fs.scaler
        # For y arrays, we use the y_feature_dict exclusively.
        indices = [feature_dict[feature] for feature in fs.feature_list if feature in feature_dict]

        # Extract the slice along the feature dimension.
        slice_ = arr[:, indices, :]  # Expected shape: (batch, num_features, seq_length)
        orig_shape = slice_.shape
        # Reshape to 2D: (batch, num_features * seq_length)
        slice_2d = slice_.reshape(slice_.shape[0], -1)
        inv_slice_2d = scaler.inverse_transform(slice_2d)
        # Reshape back to the original 3D shape.
        inv_slice = inv_slice_2d.reshape(orig_shape)
        # Update the working array.
        arr[:, indices, :] = inv_slice

    def verify_executable(self, entity, strategy_request):
        """
        Ensure that the entity contains the target array attribute.
        """
        param_config = strategy_request.param_config
        target_attr = param_config.get("target_array_attribute")
        if not target_attr:
            raise ValueError("Parameter config must include 'target_array_attribute'.")
        if not entity.has_attribute(target_attr):
            raise ValueError(f"Missing {target_attr} in dataset.")

    def get_feature_set(self, entity_id):
        """
        Retrieve a feature set entity using its entity id.
        This implementation assumes you have a strategy (e.g., GetEntityStrategy) to do so.
        """
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = GetEntityStrategy.__name__
        strategy_request.target_entity_id = entity_id

        strategy_request = self.executor_service.execute_request(strategy_request)
        self.strategy_request.add_nested_request(strategy_request)

        return strategy_request.ret_val['entity']

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': InverseScaleByFeatureSetsStrategy.__name__,
            'strategy_path': None,
            'param_config': {
                # Default target array attribute (name of the attribute holding the scaled y array)
                "target_array_attribute": "scaled_array",
                # Default output attribute for the inverse-transformed array
                "output_attribute": "inversed_scaled_array"
            }
        }


class CombineDataBundlesStrategy(DataBundleStrategy):
    name = "CombineDataBundles"
    strategy_description = 'Merges multiple DataBundleEntity instances into a unified dataset for expanded training or analysis. Validates that all input bundles have matching attribute keys, preserves feature dictionaries from the first bundle, concatenates arrays along batch dimension for X, y, row_ids, and train/test splits, creates a new DataBundleEntity with merged data, transfers child entities from the first bundle to maintain feature set relationships, and returns the combined bundle in strategy return values. Enables scaling up datasets by combining data from multiple sources or time periods.'
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, data_bundles):
        self.verify_executable(data_bundles, self.strategy_request)

        # confirm they have the same keys
        keys = data_bundles[0]._attributes.keys()
        for data_bundle in data_bundles[1:]:
            if data_bundle._attributes.keys() != keys:
                raise ValueError("DataBundles do not have the same keys")

        # combine the datasets
        combined_dataset = {}
        if data_bundles[0].has_attribute('X_feature_dict'):
            combined_dataset['X_feature_dict'] = data_bundles[0].get_attribute('X_feature_dict')
        if data_bundles[0].has_attribute('y_feature_dict'):
            combined_dataset['y_feature_dict'] = data_bundles[0].get_attribute('y_feature_dict')

        for key in keys:
            if key in ['X', 'y','row_ids', 'X_train', 'y_train', 'X_test', 'y_test']:
                combined_dataset[key] = np.concatenate([data_bundle.get_attribute(key) for data_bundle in data_bundles], axis=0)
            elif key in ['train_row_ids', 'test_row_ids', 'row_ids']:
                combined_dataset[key] = np.concatenate([data_bundle.get_attribute(key) for data_bundle in data_bundles], axis=0)

        new_bundle = DataBundleEntity()
        new_bundle.set_attributes(combined_dataset)
        old_children = data_bundles[0].children
        old_children = deepcopy(old_children)
        for child in old_children:
            new_bundle.add_child(child)

        self.strategy_request.ret_val[EntityEnum.DATA_BUNDLE.value] = new_bundle

        return self.strategy_request


    def verify_executable(self, entity, strategy_request):
        param_config = strategy_request.param_config

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CombineDataBundlesStrategy.__name__,
            'strategy_path': None,
            'param_config': {
            }
        }

