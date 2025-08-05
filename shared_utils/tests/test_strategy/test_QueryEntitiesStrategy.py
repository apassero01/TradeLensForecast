from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from django.test import TestCase
from shared_utils.strategy.QueryEntitiesStrategy import QueryEntitiesStrategy
from shared_utils.entities.Entity import Entity
from shared_utils.entities.StrategyRequestEntity import StrategyRequestEntity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy_executor.StrategyExecutor import StrategyExecutor


# Create test entity class
class TestConcreteEntity(Entity):
    entity_name = EntityEnum.ENTITY
    
    def to_db(self):
        return {}
        
    @classmethod
    def from_db(cls, data):
        return cls()


class QueryEntitiesStrategyTestCase(TestCase):
    """Test cases for QueryEntitiesStrategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy_executor = StrategyExecutor()
        
        # Create a test target entity
        self.target_entity = TestConcreteEntity()
        self.target_entity.entity_id = str(uuid.uuid4())
        
        # Create a basic strategy request
        self.strategy_request = StrategyRequestEntity()
        self.strategy_request.strategy_name = "QueryEntitiesStrategy"
        self.strategy_request.target_entity_id = self.target_entity.entity_id
        
    def test_verify_executable_with_valid_config(self):
        """Test verify_executable with valid configuration"""
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'document'
                }
            ]
        }
        
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        result = strategy.verify_executable(self.target_entity, self.strategy_request)
        
        self.assertTrue(result)
        
    def test_verify_executable_missing_filters(self):
        """Test verify_executable with missing filters parameter"""
        self.strategy_request.param_config = {}
        
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        result = strategy.verify_executable(self.target_entity, self.strategy_request)
        
        self.assertFalse(result)
        
    def test_verify_executable_with_empty_filters(self):
        """Test verify_executable with empty filters list"""
        self.strategy_request.param_config = {
            'filters': []
        }
        
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        result = strategy.verify_executable(self.target_entity, self.strategy_request)
        
        self.assertTrue(result)  # Empty filters list is valid
        
    def test_verify_executable_invalid_filter_structure(self):
        """Test verify_executable with invalid filter structure"""
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    # Missing 'operator' and 'value'
                }
            ]
        }
        
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        result = strategy.verify_executable(self.target_entity, self.strategy_request)
        
        self.assertFalse(result)
        
    def test_apply_with_single_filter(self):
        """Test apply method with a single filter"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return some entity IDs
        test_entity_ids = [str(uuid.uuid4()) for _ in range(3)]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Configure strategy request
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'document'
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result, self.strategy_request)
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 3)
        
        # Verify entity service was called correctly
        mock_entity_service.find_entities.assert_called_once_with(
            self.strategy_request.param_config['filters']
        )
        
    def test_apply_with_multiple_filters(self):
        """Test apply method with multiple filters"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return one entity ID
        test_entity_id = str(uuid.uuid4())
        mock_entity_service.find_entities.return_value = [test_entity_id]
        
        # Configure strategy request with multiple filters
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'recipe'
                },
                {
                    'attribute': 'name',
                    'operator': 'contains',
                    'value': 'chicken'
                },
                {
                    'attribute': 'creation_date',
                    'operator': 'greater_than',
                    'value': '2025-01-01'
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], [test_entity_id])
        self.assertEqual(result.ret_val['count'], 1)
        
        # Verify filters were passed correctly
        mock_entity_service.find_entities.assert_called_once_with(
            self.strategy_request.param_config['filters']
        )
        
    def test_apply_with_empty_results(self):
        """Test apply method when no entities match the filters"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return empty list
        mock_entity_service.find_entities.return_value = []
        
        # Configure strategy request
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'non_existent_type'
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], [])
        self.assertEqual(result.ret_val['count'], 0)
        
    def test_apply_with_error(self):
        """Test apply method when find_entities raises an exception"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to raise exception
        mock_entity_service.find_entities.side_effect = Exception("Database error")
        
        # Configure strategy request
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'entity_type',
                    'operator': 'equals',
                    'value': 'document'
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify error handling
        self.assertEqual(result.ret_val['error'], "Database error")
        self.assertEqual(result.ret_val['matching_entity_ids'], [])
        self.assertEqual(result.ret_val['count'], 0)
        
    def test_get_request_config(self):
        """Test get_request_config returns expected configuration"""
        config = QueryEntitiesStrategy.get_request_config()
        
        # Verify config structure
        self.assertIn('filters', config)
        self.assertIsInstance(config['filters'], list)
        self.assertGreater(len(config['filters']), 0)
        
        # Verify filter example structure
        example_filter = config['filters'][0]
        self.assertIn('attribute', example_filter)
        self.assertIn('operator', example_filter)
        self.assertIn('value', example_filter)
        
    def test_request_constructor(self):
        """Test request_constructor creates valid strategy request"""
        target_id = str(uuid.uuid4())
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'in',
                'value': ['document', 'recipe']
            }
        ]
        
        request = QueryEntitiesStrategy.request_constructor(
            target_entity_id=target_id,
            filters=filters
        )
        
        # Verify request structure
        self.assertEqual(request.strategy_name, 'QueryEntitiesStrategy')
        self.assertEqual(request.target_entity_id, target_id)
        self.assertEqual(request.param_config['filters'], filters)
        self.assertEqual(request._nested_requests, [])

    def test_case_insensitive_text_search(self):
        """Test case-insensitive text searching for contains operator"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4()) for _ in range(2)]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Configure strategy request with case-insensitive contains
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'name',
                    'operator': 'contains',
                    'value': 'task'  # Should match "Task:" in database
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 2)
        
    def test_child_ids_querying(self):
        """Test querying by child_ids attribute"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4())]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Configure strategy request to search for child_ids
        child_id = str(uuid.uuid4())
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'child_ids',
                    'operator': 'equals',
                    'value': child_id
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 1)
        
    def test_child_ids_multiple_values(self):
        """Test querying child_ids with 'in' operator"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4()) for _ in range(3)]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Configure strategy request to search for multiple child_ids
        child_ids = [str(uuid.uuid4()) for _ in range(2)]
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'child_ids',
                    'operator': 'in',
                    'value': child_ids
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 3)
        
    def test_date_comparison_different_formats(self):
        """Test date comparison with different date formats"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4()) for _ in range(2)]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Test with YYYYMMDD format
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'creation_date',
                    'operator': 'greater_than',
                    'value': '20250101'  # Should work with YYYYMMDD format
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 2)
        
    def test_date_comparison_iso_format(self):
        """Test date comparison with ISO format including time"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4())]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Test with ISO format including time
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'creation_date',
                    'operator': 'less_than',
                    'value': '2025-12-31T23:59:59.000Z'  # Should work with ISO format
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 1)
        
    def test_date_between_mixed_formats(self):
        """Test date between with mixed date formats"""
        # Create strategy
        strategy = QueryEntitiesStrategy(self.strategy_executor, self.strategy_request)
        
        # Mock the entity_service instance
        mock_entity_service = Mock()
        strategy.entity_service = mock_entity_service
        
        # Mock find_entities to return entity IDs
        test_entity_ids = [str(uuid.uuid4()) for _ in range(2)]
        mock_entity_service.find_entities.return_value = test_entity_ids
        
        # Test with mixed date formats
        self.strategy_request.param_config = {
            'filters': [
                {
                    'attribute': 'event_date',
                    'operator': 'between',
                    'value': ['20250101', '2025-12-31T00:00:00Z']  # Mixed formats
                }
            ]
        }
        
        # Execute strategy
        result = strategy.apply(self.target_entity)
        
        # Verify results
        self.assertEqual(result.ret_val['matching_entity_ids'], test_entity_ids)
        self.assertEqual(result.ret_val['count'], 2)