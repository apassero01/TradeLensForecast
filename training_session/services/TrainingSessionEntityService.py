from enum import Enum

from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity, TrainingSessionStatus
from training_session.models import TrainingSession
from training_session.strategy.TrainingSessionStrategy import CreateModelStageStrategy

DEFAULT_SESSION_UUID = '2b012d92-1fd8-4c72-a8e4-981c45a9db6b'

class TrainingSessionEntityService:
    def __init__(self):
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor_service = StrategyExecutorService(self.strategy_executor)

    def create_training_session_entity(self):
        """Create a new training session with minimal initialization"""
        self.session_model = TrainingSession()
        self.session_model.entity_id = DEFAULT_SESSION_UUID
        self.session_model.save()
        self.session = TrainingSessionEntity.from_db(self.session_model)
        self.session.strategy_history = []
        return self.session

    def initialize_params(self, X_features, y_features, sequence_set_params, start_date, end_date=None):
        self.session.X_features = X_features
        self.session.y_features = y_features
        self.session.sequence_set_params = sequence_set_params
        self.session.start_date = start_date
        return self.session
    
    def set_session(self, session_entity):
        self.session = session_entity

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
        session = TrainingSessionEntity.from_db(session)
        return session


    def execute_strat_request(self, strat_request, session_entity):

        strat_request =  self.strategy_executor_service.execute(session_entity, strat_request)
        if strat_request.add_to_history:
            session_entity.add_to_strategy_history(strat_request)


    def serialize_session(self):
        """Serialize the current session state"""

        return {
            'id': self.session.id,
            'created_at': self.session.created_at,
            'entity_map': self.session.serialize(),
            'strategy_history': [strategy_request.serialize() for strategy_request in self.session.strategy_history]
        }
    
    def save_session(self):
        model = TrainingSessionEntity.entity_to_model(self.session)
        model.save()
        return model.id



    # {
    #     "entity_name": "session"
    #     "children": [
    #         {
    #             "entity_name": "model_stage",
    #             "children": []
    #             "meta_data": {}
    #         },
    #         {
    #             "entity_name": "data_bundle",
    #             "children": []
    #             "meta_data": {}
    #         ]
    #     "meta_data": {}
    # }






