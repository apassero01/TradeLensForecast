import numpy as np

from data_bundle_manager.entities.DataBundleEntity import DataBundleEntity
from data_bundle_manager.strategy.DataBundleStrategy import SplitBundleDateStrategy, \
    ScaleByFeatureSetsStrategy, CombineDataBundlesStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy.BaseStrategy import AssignAttributesStrategy


class SequenceSetStrategy(Strategy):
    entity_type = EntityEnum.SEQUENCE_SET
    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    def apply(self, entity):
        """Apply the strategy to the entity."""
        pass

    def verify_executable(self, entity, strategy_request):
        raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

    @staticmethod
    def get_request_config():
        return {}




# class CreateDataBundleStrategy(SequenceSetStrategy):
#     entity_type = EntityEnum.SEQUENCE_SET
#     name = "CreateDataBundle"

#     def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
#         super().__init__(strategy_executor, strategy_request)
#         self.strategy_request = strategy_request
#         self.strategy_executor = strategy_executor

#     def apply(self, sequence_sets):
#         """Apply the strategy to the entity."""
#         for sequence_set in sequence_sets:
#             data_bundle = self.create_bundle(sequence_set)
#             sequence_set.set_entity_map({EntityEnum.DATA_BUNDLE.value: data_bundle})

#         return self.strategy_request

#     def create_bundle(self, sequence_sets):
#         data_bundle = DataBundleEntity()
#         return data_bundle

#     def verify_executable(self, entity, strategy_request):
#         raise NotImplementedError("Child classes must implement the 'verify_executable' method.")

#     @staticmethod
#     def get_request_config():
#         return {
#             'strategy_name': CreateDataBundleStrategy.__name__,
#             'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SET.value,
#             'param_config': {}
#         }


class PopulateDataBundleStrategy(SequenceSetStrategy):
    name = 'PopulateDataBundle'
    strategy_description = 'Transforms raw sequence data into structured 3D numpy arrays suitable for machine learning models and populates associated DataBundleEntity. Reads X_features and y_features from SequenceSetEntity, creates feature dictionaries mapping feature names to indices, extracts sequence data into X array (batch, time_steps, features) and y array (batch, features, 1) using the last timestep for targets, filters out sequences containing NaN values, generates row_ids from sequence IDs, creates separate X and y feature dictionaries, assigns all arrays and metadata to the SequenceSetEntity, then transfers this data to its child DataBundleEntity via AssignAttributesStrategy. Essential preprocessing step that converts time series sequences into ML-ready tensor formats.'
    
    def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.strategy_request = strategy_request
        self.strategy_executor = strategy_executor

    def apply(self, sequence_sets):
        if not isinstance(sequence_sets, list):
            sequence_sets = [sequence_sets]
        self.verify_executable(sequence_sets, self.strategy_request)
        for sequence_set  in sequence_sets:
            X_features = sequence_set.get_attribute('X_features')
            y_features = sequence_set.get_attribute('y_features')
            feature_dict = self.create_feature_dict(X_features, y_features)
            X, y, row_ids = self.create_3d_array_seq(sequence_set, X_features, y_features, feature_dict)
            sequence_set.set_attribute('X', X)
            sequence_set.set_attribute('y', y)
            sequence_set.set_attribute('row_ids', row_ids)

            X_feature_dict, y_feature_dict = self.create_xy_feature_dict(X_features, y_features)
            sequence_set.set_attribute('X_feature_dict', X_feature_dict)
            sequence_set.set_attribute('y_feature_dict', y_feature_dict)

            bundle_ids = self.entity_service.get_children_ids_by_type(sequence_set, EntityEnum.DATA_BUNDLE)
            if not len(bundle_ids) == 1:
                raise ValueError("SequenceSetEntity should have exactly one DataBundleEntity child")
            strategy_request = self.create_strategy_request(bundle_ids[0])
            strategy_request = self.strategy_executor.execute(sequence_set, strategy_request)
            self.strategy_request.add_nested_request(strategy_request)

        return self.strategy_request


    def create_strategy_request(self, target_entity):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = AssignAttributesStrategy.__name__
        strategy_request.param_config = {
            'assign_id': target_entity,
            'attribute_map': {
                'X' : 'X',
                'y' : 'y',
                'row_ids': 'row_ids',
                'X_feature_dict': 'X_feature_dict',
                'y_feature_dict': 'y_feature_dict',
                'seq_end_dates': 'seq_end_dates'
            }
        }

        return strategy_request


    def create_feature_dict(self, X_features, y_features):
        feature_dict = {}
        for i, feature in enumerate(X_features + y_features):
            feature_dict[feature] = i

        return feature_dict

    def create_3d_array_seq(self, sequence_set, X_features, y_features, feature_dict):
        sequence_objs = sequence_set.get_attribute('sequences')
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

    def create_xy_feature_dict(self, X_features, y_features):
        X_indices_seq = np.arange(len(X_features)).tolist()
        y_indices_seq = np.arange(len(y_features)).tolist()

        X_feature_dict = {col: index for col, index in zip(X_features, X_indices_seq)}
        y_feature_dict = {col: index for col, index in zip(y_features, y_indices_seq)}

        return X_feature_dict, y_feature_dict

    def verify_executable(self, sequence_sets, strategy_request):
        pass

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': PopulateDataBundleStrategy.__name__,
            'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SET.value,
            'param_config': {}
        }


# class SplitAllBundlesDataStrategy(SequenceSetStrategy):
#     name = 'SplitAllBundlesData'
#     def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
#         super().__init__(strategy_executor, strategy_request)

#     def apply(self, sequence_sets):
#         for sequence_set in sequence_sets:
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             dates = [sequence.end_timestamp for sequence in sequence_set.sequences]
#             split_date = self.strategy_request.param_config['split_date']
#             bundle_strategy_request = self.create_strategy_request(dates, split_date)
#             self.strategy_executor.execute(data_bundle, bundle_strategy_request)

#         return self.strategy_request

#     def create_strategy_request(self, dates, split_date):
#         strategy_request = StrategyRequestEntity()
#         strategy_request.strategy_name = SplitBundleDateStrategy.__name__
#         strategy_request.strategy_path = None
#         strategy_request.param_config = {
#             'date_list' : dates,
#             'split_date' : split_date
#         }

#         return strategy_request

#     def verify_executable(self, sequence_sets, strategy_request):
#         if 'split_date' not in strategy_request.param_config.keys():
#             raise ValueError("Missing split_date in config")
#         for sequence_set in sequence_sets:
#             try:
#                 result = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             except:
#                 raise ValueError("DataBundle not found in SequenceSetEntity")



    # @staticmethod
    # def get_request_config():
    #     return {
    #         'strategy_name': SplitAllBundlesDataStrategy.__name__,
    #         'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SETS.value,
    #         'param_config': {'split_date': None}
    #     }


# class ScaleSeqSetsByFeaturesStrategy(SequenceSetStrategy):
#     name = 'ScaleSeqSetsByFeatures'
#     def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
#         super().__init__(strategy_executor, strategy_request)

#     def apply(self, sequence_sets):
#         for sequence_set in sequence_sets:
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             bundle_strategy_request = self.create_strategy_request()
#             self.strategy_executor.execute(data_bundle, bundle_strategy_request)

#         return self.strategy_request

#     def create_strategy_request(self):
#         strategy_request = StrategyRequestEntity()
#         strategy_request.strategy_name = ScaleByFeatureSetsStrategy.__name__
#         strategy_request.strategy_path = None
#         strategy_request.param_config = {
#         }

#         return strategy_request

#     def verify_executable(self, sequence_sets, strategy_request):
#         for sequence_set in sequence_sets:
#             try:
#                 result = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             except:
#                 raise ValueError("DataBundle not found in SequenceSetEntity")

#     @staticmethod
#     def get_request_config():
#         return {
#             'strategy_name': ScaleSeqSetsByFeaturesStrategy.__name__,
#             'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SETS.value,
#             'param_config': {}

#         }

# class CombineSeqBundlesStrategy(SequenceSetStrategy):
#     name = 'CombineSeqBundles'
#     def __init__(self, strategy_executor, strategy_request: StrategyRequestEntity):
#         super().__init__(strategy_executor, strategy_request)

#     def apply(self, sequence_sets):
#         data_bundles = []
#         for sequence_set in sequence_sets:
#             data_bundle = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             data_bundles.append(data_bundle)

#         bundle_strategy_request = self.create_strategy_request()
#         bundle_strategy_request = self.strategy_executor.execute(data_bundles, bundle_strategy_request)
#         self.strategy_request.ret_val[EntityEnum.DATA_BUNDLE.value] = bundle_strategy_request.ret_val[EntityEnum.DATA_BUNDLE.value]

#         return self.strategy_request


#     def create_strategy_request(self):
#         strategy_request = StrategyRequestEntity()
#         strategy_request.strategy_name = CombineDataBundlesStrategy.__name__
#         strategy_request.strategy_path = None
#         strategy_request.param_config = {
#         }

#         return strategy_request
#     def verify_executable(self, sequence_sets, strategy_request):
#         for sequence_set in sequence_sets:
#             try:
#                 result = sequence_set.get_entity(EntityEnum.DATA_BUNDLE.value)
#             except:
#                 raise ValueError("DataBundle not found in SequenceSetEntity")

#     @staticmethod
#     def get_request_config():
#         return {
#             'strategy_name': CombineSeqBundlesStrategy.__name__,
#             'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SETS.value,
#             'param_config': {}
#         }