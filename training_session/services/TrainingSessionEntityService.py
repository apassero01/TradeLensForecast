from enum import Enum

from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
from training_session.entities.TrainingSessionEntity import TrainingSessionEntity, TrainingSessionStatus
from training_session.models import TrainingSession
from training_session.strategy.TrainingSessionStrategy import CreateModelStageStrategy

class TrainingSessionEntityService:
    def __init__(self, session):
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor_service = StrategyExecutorService(self.strategy_executor)
        self.session = session


    def create_training_session_entity(self):
        return TrainingSessionEntity()

    def initialize_params(self, X_features, y_features, model_set_configs, start_date, end_date=None):
        self.session.X_features = X_features
        self.session.y_features = y_features
        self.session.model_set_configs = model_set_configs
        self.session.start_date = start_date
        return self.session

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
        entity = self.resolve_strat_request_path(strat_request, session_entity)

        return self.strategy_executor.execute(entity, strat_request)

    def resolve_strat_request_path(self, strat_request, session_entity):
        path = strat_request.strategy_path
        if not path:
            raise ValueError('Path not found in strat request')

        path_components = path.split('.')
        path_components = path_components[1:]  # Remove the first component, which is the root entity
        num_components = len(path_components)

        if num_components == 0:
            return session_entity

        current_entity = session_entity
        for i, component in enumerate(path_components):
            current_entity = current_entity.get_entity(component)
            if not current_entity:
                raise ValueError(f'Entity not found for path {path}')

            if i == num_components - 1:
                return current_entity

        raise ValueError(f'Entity not found for path {path}')

    def serialize_entity_tree(self):
        return self.session.serialize()

    def serialize_session(self):
        return {
            'session_id': self.session.id,
            'status': TrainingSessionStatus(self.session.status).name,
            'created_at': self.session.created_at,
            'entity_map': self.serialize_entity_tree()
        }



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






