from shared_utils.strategy.BaseStrategy import Strategy
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.EnityEnum import EntityEnum
import logging

logger = logging.getLogger(__name__)


class QueryEntitiesStrategy(Strategy):
    """
    Strategy for querying entities based on configurable filters.
    Results are attached to the target entity as a new attribute.
    """
    
    entity_type = EntityEnum.ENTITY
    strategy_description = 'Queries entities based on filters and attaches results to target entity'
    
    def verify_executable(self, entity, strategy_request):
        """Verify the strategy can be executed with required parameters"""
        config = strategy_request.param_config
        
        # Check for required parameters
        if 'filters' not in config:
            logger.error("Missing required parameter: filters")
            return False
            
        if 'result_attribute' not in config:
            logger.error("Missing required parameter: result_attribute")
            return False
            
        # Validate filter structure
        filters = config.get('filters', [])
        if not isinstance(filters, list):
            logger.error("Filters must be a list")
            return False
            
        for filter_obj in filters:
            if not isinstance(filter_obj, dict):
                logger.error("Each filter must be a dictionary")
                return False
                
            required_keys = ['attribute', 'operator', 'value']
            if not all(key in filter_obj for key in required_keys):
                logger.error(f"Filter missing required keys. Required: {required_keys}")
                return False
                
        return True
    
    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Apply the query strategy to find entities matching the filters.
        
        param_config requirements:
        - filters: List of filter dictionaries with keys: attribute, operator, value
        - result_attribute: Name of attribute to store results on target entity
        """
        config = self.strategy_request.param_config
        filters = config.get('filters', [])
        result_attribute = config.get('result_attribute')
        
        try:
            # Use EntityService to find matching entities
            matching_entity_ids = self.entity_service.find_entities(filters)
            
            # Attach results to target entity
            entity.set_attribute(result_attribute, matching_entity_ids)
            
            # Save the entity with new attribute
            self.entity_service.save_entity(entity)
            
            # Store results in return value
            self.strategy_request.ret_val = {
                'matching_entity_ids': matching_entity_ids,
                'count': len(matching_entity_ids)
            }
            
            logger.info(f"Query found {len(matching_entity_ids)} matching entities")
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            self.strategy_request.ret_val = {
                'error': str(e),
                'matching_entity_ids': [],
                'count': 0
            }
            
        return self.strategy_request
    
    @staticmethod
    def get_request_config():
        """Return example configuration for this strategy"""
        return {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'document'
                },
                {
                    'attribute': 'creation_date',
                    'operator': 'greater_than', 
                    'value': '2025-01-01'
                }
            ],
            'result_attribute': 'query_results'
        }
    
    @classmethod
    def request_constructor(cls, target_entity_id, filters: list, result_attribute: str):
        """Convenience method to create a strategy request for this strategy"""
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = cls.__name__
        strategy_request.param_config = {
            'filters': filters,
            'result_attribute': result_attribute
        }
        strategy_request.target_entity_id = target_entity_id
        strategy_request._nested_requests = []
        return strategy_request