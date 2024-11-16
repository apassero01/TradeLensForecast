import time
from copy import deepcopy
from datetime import datetime
from enum import Enum
import uuid

import numpy as np

from shared_utils.strategies import ModelSetsStrategy
from training_session.models import TrainingSession
from training_session.services import StrategyService
from training_session.services.ModelSetService import ModelSetService
from training_session.services.StrategyService import ModelSetStrategyService, VizProcessingStrategyService


class TrainingSessionStatus(Enum):
    ACTIVE = 1
    INACTIVE = 2


class TrainingSessionService:
    def __init__(self):
        self.model_set_strategy_service = ModelSetStrategyService()
        self.viz_strategy_service = VizProcessingStrategyService()
        self.model_set_service = ModelSetService()

    def create_training_session(self):
        session = TrainingSession()
        session.status = TrainingSessionStatus.ACTIVE.value
        session.model_sets = []
        session.ordered_model_set_strategies = []
        session.created_at = datetime.now()
        return session

    def initialize_params(self, session, X_features, y_features, model_set_configs, start_date, end_date=None):
        session.X_features = X_features
        session.y_features = y_features
        session.X_feature_dict, session.y_feature_dict = self.create_xy_feature_dict(X_features, y_features)
        session.model_set_configs = model_set_configs
        session.start_date = start_date
        return session

    def apply_strategy(self, session, strategy_json):
        if strategy_json['config']['parent_strategy'] == 'ModelSetsStrategy':
            session, ret_val = self.model_set_strategy_service.apply_model_set_strategy(session, strategy_json)
        elif strategy_json['config']['parent_strategy'] == 'VizProcessingStrategy':
            session, ret_val = self.viz_strategy_service.apply_viz_processing_strategy(session, strategy_json)
        else:
            raise ValueError("Invalid parent strategy")
        return ret_val

    def create_xy_feature_dict(self, X_features, y_features):
        X_indices_seq = np.arange(len(X_features)).tolist()
        y_indices_seq = np.arange(len(y_features)).tolist()


        X_feature_dict = {col : index for col, index in zip(X_features, X_indices_seq)}
        y_feature_dict = {col : index for col, index in zip(y_features, y_indices_seq)}

        return X_feature_dict, y_feature_dict

    def save_session(self, session):
        session.status = TrainingSessionStatus.INACTIVE.value
        for strategy in session.ordered_model_set_strategies:
            strategy['config']['is_applied'] = False
        session.save()
        return session

    def print_session(self, session):
        print(f"Session ID: {session.id}")
        print(f"Status: {TrainingSessionStatus(session.status).name}")
        print(f"Created At: {session.created_at}")
        print(f"Ordered Model Set Strategies: {session.ordered_model_set_strategies}")
        print(f"X Features: {session.X_features}")
        print(f"Y Features: {session.y_features}")
        print(f"X Feature Dict: {session.X_feature_dict}")
        print(f"Y Feature Dict: {session.y_feature_dict}")
        print("\n")
        return session

    def get_sessions(self):
        sessions = TrainingSession.objects.all()
        ret_val = [
            {
                'id': session.id,
                'status': TrainingSessionStatus(session.status).name,
                'created_at': session.created_at
            }
            for session in sessions
        ]
        return ret_val

    def get_session(self, session_id):
        session = TrainingSession.objects.get(id=session_id)
        session.model_sets = []
        return session

    def serialize_session(self, session):
        serialized_session = {
            'session_id': session.id,
            'status': TrainingSessionStatus(session.status).name,
            'created_at': session.created_at,
            'X_features': session.X_features,
            'y_features': session.y_features,
            'X_feature_dict': session.X_feature_dict,
            'y_feature_dict': session.y_feature_dict,
            'model_set_configs': session.model_set_configs,
            'ordered_model_set_strategies': session.ordered_model_set_strategies,
            'start_date': session.start_date,
            'end_date': session.end_date,
            'model_sets': [ModelSetService.serialize_model_set_state(model_set) for model_set in session.model_sets],
            'X_train': session.X_train.shape if hasattr(session, 'X_train') else None,
            'X_test': session.X_test.shape if hasattr(session, 'X_test') else None,
            'y_train': session.y_train.shape if hasattr(session, 'y_train') else None,
            'y_test': session.y_test.shape if hasattr(session, 'y_test') else None,
        }
        return serialized_session


