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
        
        # Verify SQL query uses LOWER LIKE for case-insensitive search
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
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
        
        # Verify SQL query uses BETWEEN with improved date handling
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('BETWEEN', query)
        self.assertIn('to_date', query)  # New date handling format
        
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
        # Test starts_with (now case-insensitive)
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'name', 'operator': 'starts_with', 'value': 'Test'}
        ])
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
        self.assertEqual(params[1], 'Test%')
        
        # Test ends_with (now case-insensitive)
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'name', 'operator': 'ends_with', 'value': '.txt'}
        ])
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
        self.assertEqual(params[1], '%.txt')
        
        # Test not_equals
        query, params = self.entity_service._build_sql_query([
            {'attribute': 'status', 'operator': 'not_equals', 'value': 'draft'}
        ])
        self.assertIn('attributes->>%s != %s', query)

    def test_normalize_date_string_yyyymmdd(self):
        """Test _normalize_date_string with YYYYMMDD format"""
        result = self.entity_service._normalize_date_string('20250630')
        self.assertEqual(result, '2025-06-30')
        
    def test_normalize_date_string_iso_format(self):
        """Test _normalize_date_string with ISO format"""
        result = self.entity_service._normalize_date_string('2025-06-30T14:30:00.000Z')
        self.assertEqual(result, '2025-06-30')
        
    def test_normalize_date_string_with_space(self):
        """Test _normalize_date_string with space and time"""
        result = self.entity_service._normalize_date_string('2025-06-30 14:30:00')
        self.assertEqual(result, '2025-06-30')
        
    def test_normalize_date_string_already_normalized(self):
        """Test _normalize_date_string with already normalized date"""
        result = self.entity_service._normalize_date_string('2025-06-30')
        self.assertEqual(result, '2025-06-30')
        
    def test_normalize_date_string_empty(self):
        """Test _normalize_date_string with empty string"""
        result = self.entity_service._normalize_date_string('')
        self.assertEqual(result, '')
        
    def test_normalize_date_string_none(self):
        """Test _normalize_date_string with None"""
        result = self.entity_service._normalize_date_string(None)
        self.assertEqual(result, None)

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_case_insensitive_contains(self, mock_connection):
        """Test find_entities with case-insensitive contains"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'name',
                'operator': 'contains',
                'value': 'task'  # Should match "Task:" case-insensitively
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses LOWER for case-insensitive search
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
        self.assertEqual(params, ['name', '%task%'])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_case_insensitive_starts_with(self, mock_connection):
        """Test find_entities with case-insensitive starts_with"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'title',
                'operator': 'starts_with',
                'value': 'TASK'
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses LOWER for case-insensitive search
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
        self.assertEqual(params, ['title', 'TASK%'])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_case_insensitive_ends_with(self, mock_connection):
        """Test find_entities with case-insensitive ends_with"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'filename',
                'operator': 'ends_with',
                'value': '.TXT'
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses LOWER for case-insensitive search
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('LOWER(attributes->>%s) LIKE LOWER(%s)', query)
        self.assertEqual(params, ['filename', '%.TXT'])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_child_ids_equals(self, mock_connection):
        """Test find_entities with child_ids equals operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        child_id = str(uuid.uuid4())
        filters = [
            {
                'attribute': 'child_ids',
                'operator': 'equals',
                'value': child_id
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses JSONB ? operator for child_ids
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('child_ids ? %s', query)
        self.assertEqual(params, [child_id])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_child_ids_not_equals(self, mock_connection):
        """Test find_entities with child_ids not_equals operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        child_id = str(uuid.uuid4())
        filters = [
            {
                'attribute': 'child_ids',
                'operator': 'not_equals',
                'value': child_id
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses NOT (child_ids ? %s)
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('NOT (child_ids ? %s)', query)
        self.assertEqual(params, [child_id])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_child_ids_in_operator(self, mock_connection):
        """Test find_entities with child_ids in operator"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        child_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        filters = [
            {
                'attribute': 'child_ids',
                'operator': 'in',
                'value': child_ids
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses OR condition for multiple child_ids
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('child_ids ? %s OR child_ids ? %s', query)
        self.assertEqual(params, child_ids)

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_entity_type_case_insensitive_contains(self, mock_connection):
        """Test find_entities with entity_type case-insensitive contains"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'entity_type',
                'operator': 'contains',
                'value': 'DOC'  # Should match "document" case-insensitively
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query uses LOWER for case-insensitive search on entity_type
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('LOWER(entity_type) LIKE LOWER(%s)', query)
        self.assertEqual(params, ['%DOC%'])

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_date_yyyymmdd_format(self, mock_connection):
        """Test find_entities with YYYYMMDD date format"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'creation_date',
                'operator': 'greater_than',
                'value': '20250630'  # YYYYMMDD format
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query handles YYYYMMDD format conversion
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('SUBSTRING(attributes->>%s, 1, 4)', query)
        self.assertIn('SUBSTRING(attributes->>%s, 5, 2)', query)
        self.assertIn('SUBSTRING(attributes->>%s, 7, 2)', query)
        self.assertIn('to_date(%s, \'YYYY-MM-DD\')', query)
        # Check that the normalized date is in params
        self.assertIn('2025-06-30', params)

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_date_iso_format(self, mock_connection):
        """Test find_entities with ISO date format including time"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'creation_date',
                'operator': 'less_than',
                'value': '2025-06-30T14:30:00.000Z'  # ISO format with time
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query handles ISO format by splitting on T and space
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('SPLIT_PART(SPLIT_PART(attributes->>%s, \'T\', 1), \' \', 1)', query)
        self.assertIn('to_date(%s, \'YYYY-MM-DD\')', query)
        # Check that the normalized date is in params
        self.assertIn('2025-06-30', params)

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_date_between_mixed_formats(self, mock_connection):
        """Test find_entities with date between using mixed formats"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'event_date',
                'operator': 'between',
                'value': ['20250101', '2025-12-31T23:59:59Z']  # Mixed formats
            }
        ]
        self.entity_service.find_entities(filters)
        
        # Verify SQL query handles both date formats
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('BETWEEN to_date(%s, \'YYYY-MM-DD\') AND to_date(%s, \'YYYY-MM-DD\')', query)
        # Check that both normalized dates are in params
        self.assertIn('2025-01-01', params)  # Normalized from 20250101
        self.assertIn('2025-12-31', params)  # Normalized from 2025-12-31T23:59:59Z

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_boolean_attribute_true(self, mock_connection):
        """Test find_entities with boolean attribute equals True"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        test_id = uuid.uuid4()
        mock_cursor.fetchall.return_value = [(test_id,)]
        
        # Execute query
        filters = [
            {
                'attribute': 'random_attribute',
                'operator': 'equals',
                'value': True
            }
        ]
        result = self.entity_service.find_entities(filters)
        
        # Verify results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], str(test_id))
        
        # Verify SQL query
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('attributes->>%s = %s', query)
        self.assertEqual(params, ['random_attribute', 'true'])  # Boolean converted to lowercase string

    @patch('shared_utils.entities.service.EntityService.connection')
    def test_find_entities_boolean_attribute_false(self, mock_connection):
        """Test find_entities with boolean attribute equals False"""
        # Set up mock cursor
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        # Execute query
        filters = [
            {
                'attribute': 'is_active',
                'operator': 'equals',
                'value': False
            }
        ]
        result = self.entity_service.find_entities(filters)
        
        # Verify results
        self.assertEqual(len(result), 0)
        
        # Verify SQL query
        query, params = mock_cursor.execute.call_args[0]
        self.assertIn('attributes->>%s = %s', query)
        self.assertEqual(params, ['is_active', 'false'])  # Boolean converted to lowercase string