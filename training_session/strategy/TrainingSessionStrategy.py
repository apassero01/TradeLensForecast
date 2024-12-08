import numpy as np
import requests

from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from sequenceset_manager.models import SequenceSet, Sequence
# from sequenceset_manager.strategy.SequenceSetStrategy import CombineSeqBundlesStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy
from train_eval_manager.entities.entities import ModelStageEntity
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.models import TrainingSession

class TrainingSessionStrategy(Strategy):
    '''
    The TrainingSessionStrategy class is a concrete class for manipulating data within the TrainingSession.

    TrainingSessionStrategy components can manipulate data within the ModelStage.

    For example a TrainingSessionService could create a feature set inside the TrainingSession.
    '''
    entity_type = EntityEnum.TRAINING_SESSION
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        pass

    def verify_executable(self, session_entity, strategy_request):
        pass

class GetSequenceSetsStrategy(TrainingSessionStrategy):
    '''
    The GetSequenceSetsStrategy class is a concrete class for getting sequence sets inside the TrainingSessionEntity.
    '''
    url = 'http://localhost:8000/sequenceset_manager/get_sequence_data'
    name = "GetSequenceSets"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        self.verify_executable(session_entity, self.strategy_request)
        param_config = self.strategy_request.param_config

        features = param_config['X_features'] + param_config['y_features']
        sequence_sets = []

        for param in param_config['model_set_configs']:
            sequence_set = SequenceSetEntity()
            
            # Set attributes directly
            sequence_set.set_attribute('dataset_type', param_config['dataset_type'])
            sequence_set.set_attribute('sequence_length', param['sequence_length'])
            sequence_set.set_attribute('start_timestamp', param['start_timestamp'])
            sequence_set.set_attribute('sequences', [])
            sequence_set.set_attribute('metadata', param)
            sequence_set.set_attribute('X_features', param_config['X_features'])
            sequence_set.set_attribute('y_features', param_config['y_features'])

            param['features'] = features

            response = requests.get(self.url, params=param)
            if response.status_code == 200:
                try:
                    data = response.json()
                    for obj in data:
                        sequence_data = obj['sliced_data']
                        sequence_data_array = np.array(sequence_data)
                        # Check if sequence_data contains NaN
                        if not np.isnan(sequence_data_array).any():
                            sequence = Sequence(
                                id=obj['id'],
                                start_timestamp=obj['start_timestamp'],
                                end_timestamp=obj['end_timestamp'],
                                sequence_length=param['sequence_length'],
                                sequence_data=sequence_data
                            )
                            sequence_set.get_attribute('sequences').append(sequence)

                    # Only add sequence set if it has sequences
                    if sequence_set.get_attribute('sequences'):
                        sequence_sets.append(sequence_set)

                except Exception as e:
                    print(f"Failed to decode JSON: {e}")
                    raise e
            else:
                print(f"Failed to retrieve sequence data: {response.status_code}")
                print(response)
                raise ValueError("Failed to retrieve sequence data " + response.json()['error'])

        # Add sequence sets as children
        for sequence_set in sequence_sets:

            session_entity.add_child(sequence_set)
            
        return self.strategy_request

    def verify_executable(self, session_entity, strategy_request):
        config = strategy_request.param_config
        if 'X_features' not in config.keys() or not config['X_features']:
            raise ValueError("Missing X_features in config")
        if 'y_features' not in config.keys() or not config['y_features']:
            raise ValueError("Missing y_features in config")
        if 'model_set_configs' not in config.keys() or not config['model_set_configs']:
            raise ValueError("Missing model_set_configs in config")
        if 'dataset_type' not in config.keys():
            raise ValueError("Missing dataset_type in config")

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': GetSequenceSetsStrategy.__name__,
            'strategy_path': 'training_session',
            'param_config': {'X_features': None,
                             'y_features': None,
                             'model_set_configs': None,
                             'dataset_type': 'stock'}
        }

class CreateModelStageStrategy(TrainingSessionStrategy):
    '''
    The CreateModelStageStrategy class is a concrete class for creating a model stage inside the TrainingSessionEntity.
    '''
    name = "CreateModelStage"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        model_stage = "test_model_stage"
        model_stage_entity = ModelStageEntity(model_stage)
        
        # Add model stage as child
        session_entity.add_child(model_stage_entity)
        
        return self.strategy_request

    def verify_executable(self, session_entity, strategy_request):
        pass

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': CreateModelStageStrategy.__name__,
            'strategy_path': 'training_session',
            'param_config': {}
        }


# class GetDataBundleStrategy(TrainingSessionStrategy):
#     '''
#     The GetDataBundleStrategy class is a concrete class for getting data bundles inside the TrainingSessionEntity.
#     '''
#     name = "GetDataBundle"
#     def __init__(self, strategy_executor, strategy_request):
#         super().__init__(strategy_executor, strategy_request)

#     def apply(self, session_entity):
#         nested_request = self.strategy_request.nested_requests[0]
#         nested_entity = self.strategy_executor.resolve_strat_request_path(nested_request, session_entity)
#         nested_request = self.strategy_executor.execute(nested_entity, nested_request)
#         session_entity.set_entity_map({EntityEnum.DATA_BUNDLE.value: nested_request.ret_val[EntityEnum.DATA_BUNDLE.value]})

#         return self.strategy_request

#     def verify_executable(self, session_entity, strategy_request):
#         pass

#     @staticmethod
#     def get_request_config():
#         return {
#             'strategy_name': GetDataBundleStrategy.__name__,
#             'strategy_path': 'training_session',
#             'param_config': {},
#             'nested_requests': [
#                 {
#                     'strategy_name': CombineSeqBundlesStrategy.__name__,
#                     'strategy_path': EntityEnum.TRAINING_SESSION.value + '.' + EntityEnum.SEQUENCE_SETS.value,
#                     'param_config': {}
#                 }
#             ]
#         }


