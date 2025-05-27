from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.strategy_executor.service.strategy_directory.StrategyDirectory import StrategyDirectory
from celery import current_task
import logging

logger = logging.getLogger(__name__)



class StrategyExecutorService:
    def __init__(self):
        self.entity_service = EntityService()
        self.strategies = {}
        self.register_strategies(StrategyDirectory().get_strategy_classes())

    def execute(self, entity, strategy_request):
        """Execute a strategy on an entity after resolving its path"""

        logger.info(f"Executing strategy {strategy_request.strategy_name} on entity {entity.entity_id}")
        strategy_name = strategy_request.strategy_name
        strategy_cls = self.strategies.get(strategy_name)
        if not strategy_cls:
            raise ValueError(f"Strategy {strategy_name} is not registered.")

        # Create and execute the strategy
        strategy = strategy_cls(self, strategy_request)
        strat_request = strategy.apply(entity)  # Store the result in the variable

        if 'entity' in strat_request.ret_val:
            entity = strat_request.ret_val['entity']
        else:
            entity = self.entity_service.get_entity(entity.entity_id)

        if strat_request.add_to_history:
            entity.update_strategy_requests(strat_request)
            self.entity_service.save_entity(strat_request)

        self.entity_service.save_entity(entity)

        logger.info(f"Strategy {strategy_name} executed successfully")

        return strat_request

    def execute_request(self, strategy_request: StrategyRequestEntity, wait: bool = True):
        """
        Enqueue a strategy request as a Celery task when called from outside a task.
        If called from within a task (i.e. a nested call), run the strategy synchronously.

        If wait is True, then wait for the result (if offloaded via Celery);
        if wait is False, return immediately with the AsyncResult.
        """
        target_entity_ids = []
        if strategy_request.has_attribute('target_entity_ids'):
            target_entity_ids = strategy_request.get_attribute('target_entity_ids')

        if target_entity_ids is None:
            target_entity_ids = [strategy_request.target_entity_id]
        else :
            target_entity_ids += [strategy_request.target_entity_id]
        # Ensure target_entity_ids is a set
        target_entity_ids = set(target_entity_ids)

        ret_val = None
        for target_entity_id in target_entity_ids:
            target_entity = self.entity_service.get_entity(target_entity_id)
            if target_entity is None:
                raise ValueError(f'Entity not found for id: {target_entity_id}')
            # Check if we're already inside a Celery task.
            if self.is_running_in_task():
                # Already inside a task, so run synchronously.
                logger.info("Running strategy synchronously")
                ret_val = self.execute(target_entity, strategy_request)
            else:
                # Not inside a task; offload execution as a new Celery task.
                from shared_utils.tasks import execute_strategy_request  # Import our Celery task.
                logger.info("Offloading strategy execution to Celery")
                # If needed, serialize the strategy_request (e.g., using to_dict) for safe transport.
                task = execute_strategy_request.delay(strategy_request)
                if wait:
                    # For callers that need the result, wait for the task to complete.
                    ## TODO at some point we should be sending and forgetting, and the executors that need to wait for results should do that themselves not block other tasks
                    ret_val = task.get(timeout=600)
                else:
                    ret_val = task

        return ret_val

    @staticmethod
    def is_running_in_task():
        # current_task.request is available even when not in a task, so check its id.
        return hasattr(current_task, 'request') and getattr(current_task.request, 'id', None) is not None


    def register_strategies(self, directory):
        for entity, strategies in directory.items():
            for strategy in strategies:
                self.strategies[strategy.__name__] = strategy

    def get_registry(self):
        serialized_registry = {}
        for entity, strategies in StrategyDirectory().get_strategy_classes().items():
            if entity not in serialized_registry:
                serialized_registry[entity] = []
            serialized_registry[entity] += [s.serialize() for s in strategies]
        return serialized_registry



