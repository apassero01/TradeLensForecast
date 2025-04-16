import logging
import pydevd_pycharm

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer

from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy_executor import StrategyExecutor
from shared_utils.strategy_executor.service.StrategyExecutorService import StrategyExecutorService
logger = logging.getLogger(__name__)
# Instantiate the service with your StrategyExecutor dependency.
executor_service = StrategyExecutorService()

@shared_task
def execute_strategy_request(strategy_request):
    """
    Celery task to execute a strategy request.
    (Perform any conversion to a StrategyRequestEntity if necessary.)
    """
    # pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    # Here, we assume strategy_request_data is already a StrategyRequestEntity.
    # Retrieve the target entity within the service call.
    target_entity = executor_service.entity_service.get_entity(strategy_request.target_entity_id)
    logger.info("Task started: executing strategy request")
    result = executor_service.execute(target_entity, strategy_request)
    logger.info("Task finished: strategy execution complete")

    return result