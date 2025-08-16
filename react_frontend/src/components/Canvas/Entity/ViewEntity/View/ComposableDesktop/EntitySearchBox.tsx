import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

// Helper function to get entity icon
function getEntityIcon(type: string): string {
  const iconMap: Record<string, string> = {
    'api_model': '🤖',
    'agent': '🤖',
    'document': '📄',
    'file': '📁',
    'folder': '📁',
    'view': '👁️',
    'chat': '💬',
    'message': '💬',
    'user': '👤',
    'data': '💾',
    'model': '🧠',
    'training': '🎯',
    'session': '🔗',
    'entity': '⚡',
    'recipe': '🍳',
    'meal': '🍽️',
    'calendar': '📅',
    'event': '📅',
    'task': '✅',
    'note': '📝',
    'image': '🖼️',
    'video': '🎥',
    'audio': '🎵',
  };
  
  return iconMap[type.toLowerCase()] || '📦';
}

interface EntitySearchBoxProps {
  data?: any;
  sendStrategyRequest: (request: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface SearchResultItemProps {
  entityId: string;
  onDoubleClick: (entityId: string) => void;
  onPin?: (entityId: string) => void;
}

const SearchResultItem: React.FC<SearchResultItemProps> = ({
  entityId,
  onDoubleClick,
  onPin,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // Single click doesn't do anything special in this implementation
  }, []);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDoubleClick(entityId);
  }, [entityId, onDoubleClick]);

  const handleRightClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onPin) {
      onPin(entityId);
    }
  }, [entityId, onPin]);

  if (!node || !node.data) {
    return (
      <div className="px-2 py-1 text-xs text-gray-600">(missing)</div>
    );
  }

  const type = node?.data?.entity_type || 'entity';
  
  // Get display name
  const displayName = node?.data?.name || 
                     node?.data?.entity_name || 
                     node?.data?.display_name ||
                     node?.data?.title ||
                     node?.data?.attributes?.name ||
                     node?.data?.attribute_map?.name ||
                     node?.data?.meta_data?.name ||
                     node?.data?.entity_type || 
                     'Unnamed';

  return (
    <div
      className="flex items-center gap-1.5 py-1 px-2 rounded text-sm transition-all cursor-pointer select-none hover:bg-gray-800 text-gray-300"
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleRightClick}
      title={`${type} • ${displayName} • Double-click to open • Right-click to pin • ${entityId}`}
    >
      {/* Entity Icon */}
      <div className="flex-shrink-0 text-sm">
        {getEntityIcon(type)}
      </div>
      
      {/* Entity Name */}
      <div className="truncate min-w-0 font-medium">
        {displayName}
      </div>
      
      {/* Entity Type Badge */}
      <div className="ml-auto text-[10px] text-gray-500 bg-gray-800 px-1 rounded">
        {type}
      </div>
    </div>
  );
};

export const EntitySearchBox: React.FC<EntitySearchBoxProps> = ({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const viewEntity: any = useRecoilValue(nodeSelectorFamily(viewEntityId));

  // Extract handlers from data prop
  const onEntityDoubleClick = data?.onEntityDoubleClick || ((entityId: string) => {
    console.log('EntitySearchBox: Entity double-clicked:', entityId);
  });
  const onEntityPin = data?.onEntityPin || ((entityId: string) => {
    console.log('EntitySearchBox: Entity pinned:', entityId);
  });

  // Debounced search effect
  useEffect(() => {
    if (!searchQuery.trim()) {
      // Don't clear search results - just return and let them persist
      return;
    }

    const timeoutId = setTimeout(() => {
      if (viewEntityId) {
        // Execute three parallel queries using StrategyRequestBuilder
        // Send requests to the view entity itself, not the parent
        const nameSearchRequest = StrategyRequests.queryEntities(
          viewEntityId,
          [{ attribute: "name", operator: "contains", value: searchQuery }],
          "name_search_results",
          true // add_to_history
        );

        const typeSearchRequest = StrategyRequests.queryEntities(
          viewEntityId,
          [{ attribute: "entity_type", operator: "contains", value: searchQuery }],
          "type_search_results",
          true // add_to_history
        );

        const textSearchRequest = StrategyRequests.queryEntities(
          viewEntityId,
          [{ attribute: "text", operator: "contains", value: searchQuery }],
          "text_search_results",
          true // add_to_history
        );

        // Send all three requests in parallel
        sendStrategyRequest(nameSearchRequest);
        sendStrategyRequest(typeSearchRequest);
        sendStrategyRequest(textSearchRequest);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [searchQuery, viewEntityId, sendStrategyRequest]);

  // Process search results with prioritization and deduplication
  const searchResults = useMemo(() => {
    if (!searchQuery.trim() || !viewEntityId) return [];

    // Read directly from the view entity's data (not through parent_attributes)
    const nameResults = viewEntity?.data?.name_search_results || [];
    const typeResults = viewEntity?.data?.type_search_results || [];
    const textResults = viewEntity?.data?.text_search_results || [];

    // Combine with prioritization and deduplication
    const combined = [...new Set([...nameResults, ...typeResults, ...textResults])];
    return combined;
  }, [searchQuery, viewEntity?.data, viewEntityId]);

  return (
    <div className="flex flex-col">
      {/* Search Input */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800">
        <div className="relative">
          <input
            type="text"
            placeholder="Search entities by name, type, or text..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-2 text-sm bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Search Results */}
      {searchQuery.trim() && searchResults.length > 0 && (
        <div className="flex-shrink-0 border-b border-gray-800">
          <div className="p-2 bg-gray-800">
            <div className="text-xs text-gray-400 mb-2">
              🔍 Search Results ({searchResults.length})
            </div>
            <div className="max-h-48 overflow-auto space-y-1">
              {searchResults.map((entityId) => (
                <SearchResultItem
                  key={`search-${entityId}`}
                  entityId={entityId}
                  onDoubleClick={onEntityDoubleClick}
                  onPin={onEntityPin}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Results - No Results Found */}
      {searchQuery.trim() && searchResults.length === 0 && viewEntityId && (
        <div className="flex-shrink-0 border-b border-gray-800">
          <div className="p-3 bg-gray-800 text-center">
            <div className="text-xs text-gray-500">
              No entities found matching "{searchQuery}"
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

