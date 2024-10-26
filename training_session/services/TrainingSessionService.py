from copy import deepcopy
from enum import Enum

import numpy as np

from shared_utils.strategies import ModelSetsStrategy
from training_session.models import TrainingSession


class TrainingSessionStatus(Enum):
    ACTIVE = 1
    INACTIVE = 2


class TrainingSessionService:

    def create_training_session(self):
        session = TrainingSession()
        session.status = TrainingSessionStatus.ACTIVE.value
        session.model_sets = []
        session.ordered_model_set_strategies = []
        return session

    def initialize_params(self, session, X_features, y_features):
        session.X_features = X_features
        session.y_features = y_features
        session.X_feature_dict, session.y_feature_dict = self.create_xy_feature_dict(X_features, y_features)
        return session

    def create_xy_feature_dict(self, X_features, y_features):
        X_indices_seq = np.arange(len(X_features))
        y_indices_seq = np.arange(len(y_features))

        X_feature_dict = {col : index for col, index in zip(X_features, X_indices_seq)}
        y_feature_dict = {col : index for col, index in zip(y_features, y_indices_seq)}

        return X_feature_dict, y_feature_dict

    def apply_model_set_strategy(self, session, config):
        """
        Apply a strategy to the session object to generate a new model set.
        """

        config['step_number'] = self.get_next_strategy_number(session)
        try:
            populated_config = self.populate_strategy_config(session, config)
            strategy = ModelSetsStrategy.get_strategy_instance(populated_config)
            session.model_sets = strategy.apply(session.model_sets)
            session.ordered_model_set_strategies.append(config)
        except Exception as e:
            # Optional: Handle exceptions if needed
            raise e
        return session

    def populate_strategy_config(self, session, config):
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

    def get_next_strategy_number(self, session):
        if session.ordered_model_set_strategies:
            return len(session.ordered_model_set_strategies)
        return 0
