import numpy as np
from matplotlib.font_manager import json_load
from numpy.f2py.auxfuncs import throw_error

from sequenceset_manager.models import Sequence, SequenceSet
from shared_utils.strategies import ModelSetsStrategy
from training_session.models import ModelSet
import requests

from training_session.services.FeatureSetService import FeatureSetService


class TrainingSessionStrategy(ModelSetsStrategy):
    def __init__(self, config):
        super().__init__(config)



class RetrieveSequenceSetsStrategy(TrainingSessionStrategy):
    url = 'http://localhost:8000/sequenceset_manager/get_sequence_data'
    name = 'RetSeqSets'
    def __init__(self, config):
        super().__init__(config)
        if 'X_features' not in config.keys():
            raise ValueError("Missing X_features in config")
        if 'y_features' not in config.keys():
            raise ValueError("Missing y_features in config")
        if 'model_set_configs' not in config.keys():
            raise ValueError("Missing model_set_configs in config")
        if 'dataset_type' not in config.keys():
            raise ValueError("Missing dataset_type in config")

        self.required_keys += ['X_features', 'y_features', 'sequence_params', 'dataset_type']


    def apply(self, model_sets, **kwargs):

        features = self.config['X_features'] + self.config['y_features']
        model_sets = []

        for param in self.config['model_set_configs']:
            model_set = ModelSet()
            model_set.dataset_type = self.config['dataset_type']

            sequence_set = SequenceSet()
            sequence_set.dataset_type = self.config['dataset_type']
            sequence_set.sequence_length = param['sequence_length']
            sequence_set.start_timestamp = param['start_timestamp']
            sequence_set.sequences = []
            sequence_set.metadata = param

            param['features'] = features

            response = requests.get(self.url, params = param)
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
                            sequence_set.sequences.append(sequence)

                    # Assign model_set properties if sequences were added
                    if sequence_set.sequences:  # Only assign if sequences exist
                        model_set.data_set = sequence_set
                        model_set.X_features = self.config['X_features']
                        model_set.y_features = self.config['y_features']
                        model_sets.append(model_set)

                except Exception as e:
                    print(f"Failed to decode JSON: {e}")
                    raise e
            else :
                print(f"Failed to retrieve sequence data: {response.status_code}")
                print(response)
                raise ValueError("Failed to retrieve sequence data " + response.json()['error'] )

        return model_sets

    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'training_session',
            'type': RetrieveSequenceSetsStrategy.__name__,
            'X_features': None,
            'y_features': None,
            'model_set_configs': None,
            'dataset_type': 'stock'
        }


class CreateFeatureSetsStrategy(TrainingSessionStrategy):
    name = 'CreateFeatureSets'
    def __init__(self, config):
        super().__init__(config)
        if 'feature_set_configs' not in config.keys():
            raise ValueError("Missing feature_set_configs in config")
        for feature_set_config in config['feature_set_configs']:
            if 'feature_set_type' not in feature_set_config.keys():
                raise ValueError("Missing feature_set_type in feature_set_config")
            if 'feature_list' not in feature_set_config.keys():
                raise ValueError("Missing feature_list in feature_set_config")
            if 'scaler_config' not in feature_set_config.keys():
                raise ValueError("Missing scaler_config in feature_set_config")

        self.required_keys += ['feature_set_configs']

        self.feature_set_service = FeatureSetService()


    def apply(self, model_sets, **kwargs):
        for model_set in model_sets:
            model_set.X_feature_sets = []
            model_set.y_feature_sets = []
            model_set.Xy_feature_sets = []
            for config in self.config['feature_set_configs']:
                feature_set = self.feature_set_service.create_feature_set(config)
                if config['feature_set_type'] == 'X':
                    model_set.X_feature_sets.append(feature_set)
                elif config['feature_set_type'] == 'y':
                    model_set.y_feature_sets.append(feature_set)
                elif config['feature_set_type'] == 'Xy':
                    model_set.Xy_feature_sets.append(feature_set)
                else:
                    raise ValueError("Invalid feature_set_type")
        return model_sets

    @staticmethod
    def get_default_config():
        return {
            'parent_strategy': 'ModelSetsStrategy',
            'm_service': 'training_session',
            'type': CreateFeatureSetsStrategy.__name__,
            'feature_set_configs': [
                {
                    'feature_set_type': 'X',
                    'feature_list': [],
                    'scaler_config': {}
                },
            ]
        }





