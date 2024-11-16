from copy import deepcopy

import numpy as np
import importlib

from django.apps import apps
import requests
from numpy.f2py.auxfuncs import throw_error

from shared_utils.strategies import ModelSetsStrategy
from training_session.VizProcessingStrategies import HistVizProcessingStrategy, LineVizProcessingStrategy, \
    SequenceVizProcessingStrategy
from training_session.services.ModelSetService import ModelSetService
from training_session.strategies import RetrieveSequenceSetsStrategy, CreateFeatureSetsStrategy
from preprocessing_manager.strategies import Create3dArraySequenceSetStrategy, TrainTestSplitDateStrategy, ScaleByFeaturesStrategy, CombineDataSetsStrategy

class StrategyService:
    @staticmethod
    def populate_strategy_config(session, config):
        """
        Populate the config dictionary with values from the session if any keys have a value of None.
        """
        populated_config = deepcopy(config)
        for key in populated_config.keys():
            if populated_config[key] is None:
                # Check if the session object has an attribute with the same name as the key
                if hasattr(session, key):
                    # Assign the value from the session attribute to the config dictionary
                    populated_config[key] = getattr(session, key)
                else:
                    # Optional: Handle missing attributes if needed
                    raise AttributeError(f"Session does not have an attribute named '{key}'")
        return populated_config



class ModelSetStrategyService(StrategyService):
    registered_strategies = [
        RetrieveSequenceSetsStrategy,
        CreateFeatureSetsStrategy,
        Create3dArraySequenceSetStrategy,
        TrainTestSplitDateStrategy,
        ScaleByFeaturesStrategy,
        CombineDataSetsStrategy
    ]

    @staticmethod
    def apply_model_set_strategy(session, strategy_json):
        """
        Apply a strategy to the session object to generate a new model set.
        """
        if session.model_sets is None:
            session.model_sets = []

        print(session.ordered_model_set_strategies)
        config = strategy_json["config"]
        amend_val = ModelSetStrategyService.strategy_hist_diff(session.ordered_model_set_strategies, strategy_json)
        if amend_val is not None:
            config['step_number'] = amend_val
        else:
            config['step_number'] = ModelSetStrategyService.get_next_strategy_number(session)
        try:
            populated_config = ModelSetStrategyService.populate_strategy_config(session, config)
            strategy = ModelSetStrategyService.get_strategy_instance(populated_config)

            if 'is_final' in populated_config.keys() and populated_config['is_final']:
                session.X_train, session.X_test, session.y_train, session.y_test, session.train_row_ids, session.test_row_ids = strategy.apply(session.model_sets)
            else:
                session.model_sets = strategy.apply(session.model_sets)

            strategy.config['is_applied'] = True
            strategy_json['config'] = strategy.config
            if amend_val is not None:
                session.ordered_model_set_strategies = ModelSetStrategyService.amend_strategy_hist(session.ordered_model_set_strategies, strategy_json, amend_val)
            else:
                session.ordered_model_set_strategies.append(strategy_json)

            print(session.ordered_model_set_strategies)

        except Exception as e:
            print(f"Failed to apply strategy: {e}")
            raise e

        return session, None

    @staticmethod
    def get_next_strategy_number(session):
        if session.ordered_model_set_strategies:
            return len(session.ordered_model_set_strategies)
        return 0

    @staticmethod
    def get_available_strategies():
        available_stragies = []
        for strategy in ModelSetStrategyService.registered_strategies:
            available_stragies.append({'name': strategy.name, 'config': strategy.get_default_config()})
        return available_stragies

    @staticmethod
    def strategy_hist_diff(ordered_stategies, new_strategy_json):
        """
        Compare the new strategy with the ordered strategies to determine if the new strategy is different.
        """
        for i,strategy in enumerate(ordered_stategies):
            if strategy['name'] == new_strategy_json['name']:
                return strategy['config']['step_number']

        return None

    @staticmethod
    def amend_strategy_hist(ordered_stategies, new_strategy_json, step_number):
        """
        Amend the strategy history with the new strategy.
        """
        for i,strategy in enumerate(ordered_stategies):
            if strategy['config']['step_number'] == step_number:
                ordered_stategies[i] = new_strategy_json
            if strategy['config']['step_number'] > step_number:
                ordered_stategies[i]['config']['is_applied'] = False
        return ordered_stategies

    @staticmethod
    def get_strategy_instance(config):
        try:
            app_config = apps.get_app_config(config['m_service'])

            strategies_module = importlib.import_module(f"{app_config.name}.strategies")

            strategy_class = getattr(strategies_module, config['type'])

            return strategy_class(config)
        except Exception as e:
            raise ImportError(f"Error loading strategy '{config['type']}' "
                              f"from app '{config['m_service']}': {str(e)}")

class VizProcessingStrategyService(StrategyService):
    registered_strategies = [
        HistVizProcessingStrategy,
        LineVizProcessingStrategy,
        SequenceVizProcessingStrategy
    ]

    @staticmethod
    def apply_viz_processing_strategy(session, strategy_json):
        """
        Apply a visualization processing strategy to the session object.
        """

        config = strategy_json["config"]
        if 'data_selection' not in config.keys():
            raise ValueError("Missing 'data_selection' in config")

        data_set, ids = VizProcessingStrategyService.get_data_set(session, config['data_selection'])
        print(f'{config['data_selection']} - {data_set.shape}')

        if 'index_list' in config.keys():
            ids = ids[config['index_list']]
            sequence_meta_data = VizProcessingStrategyService.get_sequence_metadata(ids)
            data_set = data_set[config['index_list']]
            # sequence_set, sequence_meta_data = ModelSetService.get_sequence_set_metadata_by_id(session.model_sets[0], ids)
            config['data_set_metadata'] = sequence_meta_data

        config['m_service'] = 'training_session'
        config = VizProcessingStrategyService.populate_strategy_config(session, config)
        strategy = VizProcessingStrategyService.get_strategy_instance(config)

        result = strategy.apply(data_set)

        return session, result

    @staticmethod
    def get_data_set(session, data_selection):
        '''
        Get the data set based on the data selection
        essentially parsing the front end representation of this item we need to extract the data from.
        #TODO - in the future we can have more shared state models between the front end and back end
        '''
        if data_selection['model_set_X_train']:
            ret_list = np.concatenate([model_set.X_train for model_set in session.model_sets])
            ids = session.train_row_ids
        elif data_selection['model_set_X_test']:
            ret_list =  np.concatenate([model_set.X_test for model_set in session.model_sets])
            ids = session.test_row_ids
        elif data_selection['model_set_y_train']:
            ret_list = np.concatenate([model_set.y_train for model_set in session.model_sets])
            ids = session.train_row_ids
        elif data_selection['model_set_y_test']:
            ret_list = np.concatenate([model_set.y_test for model_set in session.model_sets])
            ids = session.test_row_ids
        elif data_selection['session_X_train']:
            ret_list = session.X_train
            ids = session.train_row_ids
        elif data_selection['session_X_test']:
            ret_list = session.X_test
            ids = session.test_row_ids
        elif data_selection['session_y_train']:
            ret_list = session.y_train
            ids = session.train_row_ids
        elif data_selection['session_y_test']:
            ret_list = session.y_test
            ids = session.test_row_ids
        else:
            raise ValueError("Invalid data selection")

        return ret_list, ids

    @staticmethod
    def get_sequence_metadata(sequence_ids):
        response = requests.get('http://localhost:8000/sequenceset_manager/get_sequence_metadata_by_ids/', params={'ids': sequence_ids})
        if response.status_code == 200:
            return response.json()
        else:
            throw_error(f"Failed to get sequence metadata: {response.text}")


    @staticmethod
    def get_available_strategies():
        available_stragies = []
        for strategy in VizProcessingStrategyService.registered_strategies:
            available_stragies.append({'name': strategy.name, 'config': strategy.get_default_config()})
        return available_stragies

    @staticmethod
    def get_strategy_instance(config):
        try:
            app_config = apps.get_app_config(config['m_service'])

            strategies_module = importlib.import_module(f"{app_config.name}.VizProcessingStrategies")

            strategy_class = getattr(strategies_module, config['type'])

            return strategy_class(config)
        except Exception as e:
            raise ImportError(f"Error loading strategy '{config['type']}' "
                              f"from app '{config['m_service']}': {str(e)}")



