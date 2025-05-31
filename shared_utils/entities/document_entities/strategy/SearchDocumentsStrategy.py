import logging
from typing import List, Dict, Any, Optional

from shared_utils.entities.Entity import Entity
from shared_utils.entities.EnityEnum import EntityEnum
from shared_utils.strategy.BaseStrategy import Strategy, StrategyRequestEntity


class SearchDocumentsStrategy(Strategy):
    """
    Searches for DocumentEntity instances by name, docName, or text content within a given scope.
    """
    strategy_name = 'SearchDocumentsStrategy'
    strategy_description = 'Searches for DocumentEntity instances by name, docName, or text content'
    config_attributes = ['query', 'search_type', 'scope_entity_id']
    can_handle_stale_entity = True

    def validate_executable(self, entity: Entity, strategy_request: StrategyRequestEntity) -> bool:
        """
        Validate that the strategy can be executed.
        """
        config = strategy_request.param_config
        
        if not config.get('query'):
            logging.error("Query is required for SearchDocumentsStrategy")
            return False
        
        search_type = config.get('search_type', 'name')
        if search_type not in ['name', 'docName', 'content']:
            logging.error(f"Invalid search_type: {search_type}")
            return False
        
        return True

    def _search_descendants(self, entity: Entity, query: str, search_type: str, results: List[Dict[str, Any]], path: List[str]):
        """
        Recursively search all descendant DocumentEntity instances.
        """
        # Check if this entity matches the search criteria
        if entity.entity_name == EntityEnum.DOCUMENT:
            match = False
            
            if search_type == 'name':
                name = entity.get_attribute('name', '')
                if query.lower() in name.lower():
                    match = True
            elif search_type == 'docName':
                doc_name = entity.get_attribute('docName', '')
                if doc_name and query.lower() in doc_name.lower():
                    match = True
            elif search_type == 'content':
                text = entity.get_attribute('text', '')
                if query.lower() in text.lower():
                    match = True
            
            if match:
                # Build the path string
                path_str = ' / '.join(path[1:]) if len(path) > 1 else '/'  # Skip root in path
                
                results.append({
                    'entity_id': entity.entity_id,
                    'name': entity.get_attribute('name', 'Unnamed'),
                    'docName': entity.get_attribute('docName', ''),
                    'is_folder': entity.get_attribute('is_folder', False),
                    'file_type': entity.get_attribute('file_type', 'text'),
                    'path': path_str,
                    'match_type': search_type,
                    'snippet': self._get_snippet(entity.get_attribute('text'), query) if search_type == 'content' else None
                })
        
        # Search children
        children_ids = entity.children_ids
        for child_id in children_ids:
            child = self.entity_service.get_entity(child_id)
            if child:
                self._search_descendants(
                    child, 
                    query, 
                    search_type, 
                    results, 
                    path + [entity.get_attribute('name')]
                )

    def _get_snippet(self, text: str, query: str, context_length: int = 50) -> str:
        """
        Extract a snippet of text around the query match.
        """
        query_lower = query.lower()
        text_lower = text.lower()
        
        index = text_lower.find(query_lower)
        if index == -1:
            return ""
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(query) + context_length)
        
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet

    def apply(self, entity: Entity) -> StrategyRequestEntity:
        """
        Search for DocumentEntity instances matching the query.
        """
        self.verify_executable(entity, self.strategy_request)
        config = self.strategy_request.param_config
        
        query = config.get('query')
        search_type = config.get('search_type', 'name')
        scope_entity_id = config.get('scope_entity_id')
        
        # Determine the search scope
        if scope_entity_id:
            scope_entity = self.entity_service.get_entity(scope_entity_id)
            if not scope_entity:
                raise ValueError(f"Scope entity not found: {scope_entity_id}")
        else:
            scope_entity = entity
        
        # Perform the search
        results = []
        self._search_descendants(scope_entity, query, search_type, results, [])
        
        # Sort results by relevance (exact matches first, then partial matches)
        results.sort(key=lambda r: (
            not (query.lower() == r['name'].lower() or query.lower() == r.get('docName', '').lower()),
            r['path']
        ))
        
        # Store results
        self.strategy_request.ret_val['results'] = results
        self.strategy_request.ret_val['total_results'] = len(results)
        self.strategy_request.ret_val['query'] = query
        self.strategy_request.ret_val['search_type'] = search_type
        
        return self.strategy_request

    @staticmethod
    def get_request_config():
        """
        Default config for SearchDocumentsStrategy.
        """
        return {
            "strategy_name": "SearchDocumentsStrategy",
            "strategy_path": None,
            "param_config": {
                "query": "",
                "search_type": "name",
                "scope_entity_id": None
            }
        } 