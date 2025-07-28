from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid

from django.test import TestCase
from shared_utils.entities.service.EntityService import EntityService
from shared_utils.entities.EntityModel import EntityModel


class EntityServiceFindEntitiesTestCase(TestCase):
    """Test cases for EntityService.find_entities method"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.entity_service = EntityService()
        
    def test_find_entities_empty_filters(self):
        """Test find_entities returns empty list for empty filters"""
        result = self.entity_service.find_entities([])
        self.assertEqual(result, [])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_single_filter_entity_type(self, mock_connection):
        """Test find_entities with single entity_type filter"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        test_ids = [uuid.uuid4(), uuid.uuid4()]
        mock_cursor.fetchall.return_value = [(test_ids[0],), (test_ids[1],)]
        
        # Execute query
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'equals',
                'value': 'document'
            }
        ]
        result = self.entity_service.find_entities(filters)
        
        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], str(test_ids[0]))
        self.assertEqual(result[1], str(test_ids[1]))
        
        # Verify SQL query
        mock_cursor.execute.assert_called_once()
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('SELECT entity_id FROM shared_utils_entitymodel WHERE', query)
        self.assertIn('entity_type = %s', query)
        self.assertEqual(params, ['document'])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_jsonb_attribute_equals(self, mock_connection):
        """Test find_entities with JSONB attribute equals filter"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        test_id = uuid.uuid4()
        mock_cursor.fetchall.return_value = [(test_id,)]
        
        # Execute query
        filters = [
            {
                'attribute': 'name',
                'operator': 'equals',
                'value': 'My Document'
            }
        ]
        result = self.entity_service.find_entities(filters)
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], str(test_id))
        
        # Verify SQL query
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('attributes->>%s = %s', query)
        self.assertEqual(params, ['name', 'My Document'])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_multiple_filters(self, mock_connection):
        """Test find_entities with multiple filters combined with AND"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query with multiple filters
        filters = [
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
                'attribute': 'difficulty',
                'operator': 'equals',
                'value': 'easy'
            }
        ]
        result = self.entity_service.find_entities(filters)
        
        # Verify SQL query contains AND
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn(' AND ', query)
        # Count ANDs - should be 2 for 3 conditions
        self.assertEqual(query.count(' AND '), 2)
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_contains_operator(self, mock_connection):
        """Test find_entities with contains operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'description',
                'operator': 'contains',
                'value': 'delicious'
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses LIKE with wildcards
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('attributes->>%s LIKE %s', query)
        self.assertEqual(params, ['description', '%delicious%'])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_numeric_comparison(self, mock_connection):
        """Test find_entities with numeric greater_than operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'price',
                'operator': 'greater_than',
                'value': 100
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses CASE for type detection
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('CASE', query)
        self.assertIn('::numeric >', query)
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_between_operator(self, mock_connection):
        """Test find_entities with between operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'creation_date',
                'operator': 'between',
                'value': ['2025-01-01', '2025-12-31']
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses BETWEEN
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('BETWEEN', query)
        self.assertIn('::date', query)
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_in_operator_for_entity_type(self, mock_connection):
        """Test find_entities with IN operator for entity_type"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'in',
                'value': ['document', 'recipe', 'meal_plan']
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses IN
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('entity_type IN (%s, %s, %s)', query)
        self.assertEqual(params, ['document', 'recipe', 'meal_plan'])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_in_operator_for_jsonb(self, mock_connection):
        """Test find_entities with IN operator for JSONB attribute"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'tags',
                'operator': 'in',
                'value': ['python', 'javascript', 'rust']
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('attributes->>%s IN (%s, %s, %s)', query)
        self.assertEqual(params, ['tags', 'python', 'javascript', 'rust'])
        
    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_error_handling(self, mock_connection):
        """Test find_entities error handling"""
        # Set up mock cursor to raise exception
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("Database connection error")
        
        # Execute query and expect exception
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'equals',
                'value': 'document'
            }
        ]
        
        with self.assertRaises(Exception) as context:
            self.entity_service.find_entities(filters)
            
        self.assertIn("Database connection error", str(context.exception))
        
    def test_build_sql_query_structure(self):
        """Test _build_sql_query returns correct structure"""
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'equals',
                'value': 'document'
            }
        ]
        
        query, params = self.entity_service._build_sql_query(filters)
        
        # Verify query structure
        self.assertIsInstance(query, str)
        self.assertIsInstance(params, list)
        self.assertTrue(query.startswith('SELECT entity_id FROM shared_utils_entitymodel WHERE'))
        self.assertEqual(len(params), 1)
        
    def test_build_sql_query_special_operators(self):
        """Test _build_sql_query handles all special operators"""
        # Test starts_with
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'name', 'operator': 'starts_with', 'value': 'Test'}
        ])
        self.assertIn('attributes->>%s LIKE %s', query)
        self.assertEqual(params[1], 'Test%')
        
        # Test ends_with
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'name', 'operator': 'ends_with', 'value': '.txt'}
        ])
        self.assertIn('attributes->>%s LIKE %s', query)
        self.assertEqual(params[1], '%.txt')
        
        # Test not_equals
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'status', 'operator': 'not_equals', 'value': 'draft'}
        ])
        self.assertIn('attributes->>%s != %s', query)