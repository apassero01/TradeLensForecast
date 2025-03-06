from shared_utils.strategy_executor import StrategyExecutor
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.service.strategy_directory.StrategyDirectory import StrategyDirectory


class StrategyExecutorService:
    def __init__(self, strategy_executor: StrategyExecutor):
        self.strategy_executor = strategy_executor
        self.entity_service = EntityService()
        self.register_strategies(StrategyDirectory().get_strategy_classes())


    def execute(self, entity, strategy_request):
        """Execute a strategy on an entity after resolving its path"""
        strat_request = self.strategy_executor.execute(entity, strategy_request)

        if 'entity' in strat_request.ret_val:
            entity = strat_request.ret_val['entity']

        if strat_request.add_to_history:
            entity.update_strategy_requests(strat_request)
            self.entity_service.save_entity(strat_request)


        self.entity_service.save_entity(entity)

        return strat_request

    def execute_request(self, strategy_request: StrategyRequestEntity):
        '''
        Execute a strategy request
        '''
        target_entity = self.entity_service.get_entity(strategy_request.target_entity_id)
        if target_entity is None:
            raise ValueError(f'Entity not found for id: {strategy_request.target_entity_id}')

        strategy_request = self.execute(target_entity, strategy_request)

        return strategy_request


    def register_strategies(self, directory):
        for entity, strategies in directory.items():
            for strategy in strategies:
                self.strategy_executor.register_strategy(strategy.__name__, strategy)

    def get_registry(self):
        serialized_registry = {}
        for entity, strategies in StrategyDirectory().get_strategy_classes().items():
            if entity not in serialized_registry:
                serialized_registry[entity] = []
            serialized_registry[entity] += [s.serialize() for s in strategies]
        return serialized_registry



