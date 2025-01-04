from enum import Enum

from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity, TrainingSessionStatus
from training_session.models import TrainingSession

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
    
    def set_session(self, session_entity):
        self.session = session_entity

    def get_session(self, session_id):
        session = TrainingSession.objects.get(id=session_id)
        session = TrainingSessionEntity.from_db(session)
        return session

    def execute_strat_request(self, new_strat_request, session_entity):

        new_strat_request =  self.strategy_executor_service.execute(session_entity, new_strat_request)
        # if strat_request.add_to_history: #TODO maybe add this back in
        for i, strategy_request in enumerate(session_entity.strategy_history):
            if strategy_request.entity_id == new_strat_request.entity_id:
                session_entity.strategy_history[i] = new_strat_request
                return
        session_entity.add_to_strategy_history(new_strat_request)

        #TODO later we want to be able to potentially change the status of downstream requests when an existing request is re-executed


    def serialize_session(self):
        """Serialize the current session state"""

        return {
            'id': self.session.id,
            'created_at': self.session.created_at,
            'entity_map': self.session.serialize(),
            'strategy_history': [strategy_request.serialize() for strategy_request in self.session.strategy_history]
        }
    
    def save_session(self):
        model = TrainingSessionEntity.to_db(self.session)
        model.save()
        return model.id







