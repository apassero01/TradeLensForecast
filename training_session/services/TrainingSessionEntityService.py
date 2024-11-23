from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
from training_session.strategy.TrainingSessionStrategy import CreateModelStageStrategy


class TrainingSessionEntityService:
    def __init__(self):
        self.strategy_executor = StrategyExecutor()
        self.strategy_executor.register_strategy(CreateModelStageStrategy.__name__, CreateModelStageStrategy)


    def execute_strat_request(self, strat_request, session_entity):
        entity = self.resolve_strat_request_path(strat_request, session_entity)

        self.strategy_executor.execute(entity, strat_request)

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






