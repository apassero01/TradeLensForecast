from shared_utils.strategy.BaseStrategy import Strategy, SetAttributesStrategy
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
    strategy_description = '''Queries entities based on filters and attaches results to target entity.
    
    Usage:
    - filters: List of filter objects with keys: attribute, operator, value
    - result_attribute_name: (Optional) Name of attribute to store results on target entity
    
    Supported operators:
    - equals: Exact match (e.g., entity_type = "document")
    - not_equals: Not equal to value
    - contains: Text contains substring (case-sensitive)
    - starts_with: Text starts with prefix
    - ends_with: Text ends with suffix
    - greater_than: Numeric/date comparison (auto-detects type)
    - less_than: Numeric/date comparison (auto-detects type)
    - between: Value within range [min, max] (e.g., [10, 50] or ["2025-01-01", "2025-12-31"])
    - in: Value is in list of options (e.g., ["document", "recipe"])
    
    Example:
    {
        "strategy_name": "QueryEntitiesStrategy",
        "target_entity_id": "some_entity_id",
        "param_config": {
            "filters": [
                {"attribute": "entity_type", "operator": "equals", "value": "recipe"},
                {"attribute": "creation_date", "operator": "greater_than", "value": "2025-06-21"},
                {"attribute": "name", "operator": "contains", "value": "chicken"}
            ],
            "result_attribute_name": "search_results"
        }
    }
    
    Notes: Currently only supports AND logic between filters so do not try OR. 
    
    '''
    
    def verify_executable(self, entity, strategy_request):
        """Verify the strategy can be executed with required parameters"""
        config = strategy_request.param_config
        
        # Check for required parameters
        if 'filters' not in config:
            logger.error("Missing required parameter: filters")
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
        - result_attribute_name: (Optional) Name of attribute to store results on target entity
        """
        config = self.strategy_request.param_config
        filters = config.get('filters', [])
        result_attribute_name = config.get('result_attribute_name')
        
        try:
            # Use EntityService to find matching entities
            matching_entity_ids = self.entity_service.find_entities(filters)
            
            # Store results in return value
            self.strategy_request.ret_val = {
                'matching_entity_ids': matching_entity_ids,
                'count': len(matching_entity_ids)
            }
            
            logger.info(f"Query found {len(matching_entity_ids)} matching entities")
            
            # If result_attribute_name is provided, store results on target entity
            if result_attribute_name:
                try:
                    # Create a SetAttributesStrategy request to store results
                    set_attributes_request = SetAttributesStrategy.request_constructor(
                        target_entity_id=self.strategy_request.target_entity_id,
                        attribute_map={result_attribute_name: matching_entity_ids}
                    )
                    
                    # Execute the SetAttributesStrategy
                    set_strategy = SetAttributesStrategy(self.strategy_executor, set_attributes_request)
                    set_strategy.apply(entity)
                    
                    logger.info(f"Stored results in attribute '{result_attribute_name}' on entity {entity.entity_id}")
                    
                except Exception as attr_error:
                    logger.error(f"Error storing results in attribute: {str(attr_error)}")
            
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
        }
    
    @classmethod
    def request_constructor(cls, target_entity_id, filters: list):
        """Convenience method to create a strategy request for this strategy"""
        strategy_request = StrategyRequestEntity()
        strategy_request.strategy_name = cls.__name__
        strategy_request.param_config = {
            'filters': filters,
        }
        strategy_request.target_entity_id = target_entity_id
        strategy_request._nested_requests = []
        return strategy_request