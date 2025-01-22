import numpy as np
import requests

from sequenceset_manager.entities.SequenceSetEntity import SequenceSetEntity
from sequenceset_manager.models import SequenceSet, Sequence
# from sequenceset_manager.strategy.SequenceSetStrategy import CombineSeqBundlesStrategy
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy, CreateEntityStrategy
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity
from training_session.models import TrainingSession
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity

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

    Again the key is the strategy is responsible for its own implementation the framework provides a way for a strategy to be executed 
    as well as any nested strategies that need to be executed. But it is up to the strategy to determine how to execute the requests. 

    In this case, this strategy creates a sequence set for each model set config. One key rule is that in order to create a new entity, we 
    must call the CreateEntityStrategy because of how we handle the creation of entities (ie maintaining constant UUIDs through history)

    So it would be cleaner to make an initial call to CreateEntityStrategy and then separately populate it with the sequence set data and do 
    this for every model set config (model_set_configs really a misnomer). But we can just choose to combine a lot of that logic into a single strategy
    because it is easier to manage. If later it is not abstract enough, we just have to implement new strategies to hold the abstracted logic. 

    Th key point is the strategy is responsible. So in this case, if there are no nested requests (happens the first time) then we created the nested 
    requests and execute them. if there are nested requests, this means that we are recreating from some history, we just need to execute them. 
    '''
    url = 'http://localhost:8000/sequenceset_manager/get_sequence_data'
    name = "GetSequenceSets"
    def __init__(self, strategy_executor, strategy_request):
        super().__init__(strategy_executor, strategy_request)

    def apply(self, session_entity):
        self.verify_executable(session_entity, self.strategy_request)
        param_config = self.strategy_request.param_config

        features = param_config['X_features'] + param_config['y_features']

        nested_requests = self.strategy_request.get_nested_requests()

        if len(nested_requests) == 0:
            nested_requests = [self.create_sequence_set_requests(session_entity) for _ in param_config['model_set_configs']]
            self.strategy_request.add_nested_requests(nested_requests)
        else: 
            if len(nested_requests) != len(param_config['model_set_configs']):
                raise ValueError("Number of nested requests does not match number of model set configs")

        for i, param in enumerate(param_config['model_set_configs']):
            nested_request = nested_requests[i]
            # It is possible on recreating the history here, the order of the nested requests is not the same as the order of the model set configs on original creation. 
            # It could be a problem I don't think it is but again the strategy is responsible so if its broken only the strategy is responsible. 
            nested_request = self.strategy_executor.execute(session_entity, nested_request)
            sequence_set = nested_request.ret_val['child_entity']
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
                        sequence_set.set_attribute('seq_end_dates', [sequence.end_timestamp for sequence in sequence_set.get_attribute('sequences')])

                    self.entity_service.save_entity(sequence_set)

                except Exception as e:
                    print(f"Failed to decode JSON: {e}")
                    raise e
            else:
                print(f"Failed to retrieve sequence data: {response.status_code}")
                print(response)
                raise ValueError("Failed to retrieve sequence data " + response.json()['error'])

            
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
        
    def create_sequence_set_requests(self, session_entity): 
        child_request = StrategyRequestEntity() 
        child_request.strategy_name = CreateEntityStrategy.__name__
        child_request.target_entity_id = session_entity.entity_id
        child_request.param_config = {
                'entity_class': "sequenceset_manager.entities.SequenceSetEntity.SequenceSetEntity",
                'entity_uuid': None
        }
        return child_request

    @staticmethod
    def get_request_config():
        return {
            'strategy_name': GetSequenceSetsStrategy.__name__,
            'strategy_path': 'training_session',
            'param_config': {'X_features': None,
                             'y_features': None,
                             'model_set_configs': None,
                             'dataset_type': 'stock'},
            'nested_requests': [
            ]
        }




