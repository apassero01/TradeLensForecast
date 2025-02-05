from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor
import numpy as np
from shared_utils.entities.InputEntity import InputEntity
from shared_utils.strategy.BaseStrategy import GetAttributesStrategy


class StrategyRequestStrategy(Strategy):
    entity_type = EntityEnum.INPUT
    def __init__(self, strategy_executor: StrategyExecutor, strategy_request: StrategyRequestEntity):
        super().__init__(strategy_executor, strategy_request)
        self.input_type = None  

    def apply(self, entity: StrategyRequestEntity):
        
        input_children = self.entity_service.get_children(entity.entity_id, EntityEnum.INPUT)
        for input_child in input_children:
            get_input_request = self.get_input_data(input_child)

            if get_input_request.ret_val['is_param_config']:
                entity.param_config[get_input_request.ret_val['input_field_name']] = get_input_request.ret_val['data']
            else:
                entity.set_attribute(get_input_request.ret_val['input_field_name'], get_input_request.ret_val['data'])

        return self.strategy_request

        
        
    def verify_executable(self, entity: Entity, strategy_request: StrategyRequestEntity):
        return True
    
    def get_input_data(self, input_entity: InputEntity):
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = GetAttributesStrategy.__name__
        strategy_request.target_entity_id = input_entity.entity_id
        strategy_request.param_config = {
            'attribute_names': [
                'data',
                'input_field_name',
                'is_param_config'
            ]
        }
        return self.executor_service.execute_request(strategy_request)

    @staticmethod
    def get_request_config():
        return {
        }