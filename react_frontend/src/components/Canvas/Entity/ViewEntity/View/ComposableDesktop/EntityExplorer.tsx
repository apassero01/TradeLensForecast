import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface EntityExplorerProps {
  rootEntityId: string;
  onEntityDoubleClick: (entityId: string) => void;
  onEntityPin?: (entityId: string) => void;
  sendStrategyRequest: (request: any) => void;
  viewEntityId?: string;
}

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

interface EntityItemProps {
  entityId: string;
  onDoubleClick: (entityId: string) => void;
  onPin?: (entityId: string) => void;
  expandedEntities: Set<string>;
  onToggleExpanded: (entityId: string) => void;
  depth?: number;
}

const EntityItem: React.FC<EntityItemProps> = ({
  entityId,
  onDoubleClick,
  onPin,
  expandedEntities,
  onToggleExpanded,
  depth = 0,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);

  // Define all hooks before any early returns
  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // Single click doesn't do anything special in this implementation
  }, []);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDoubleClick(entityId);
  }, [entityId, onDoubleClick]);

  const handleExpandClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    const hasChildren = node?.data?.child_ids && node?.data?.child_ids.length > 0;
    if (hasChildren) {
      onToggleExpanded(entityId);
    }
  }, [entityId, node?.data?.child_ids, onToggleExpanded]);

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
  const hasChildren = node?.data?.child_ids && node?.data?.child_ids.length > 0;
  const isExpanded = expandedEntities.has(entityId);
  
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

  // Get children entities if expanded
  const children = hasChildren && isExpanded 
    ? allEntities.filter(entity => node.data.child_ids.includes(entity.data.entity_id))
    : [];

  const indentStyle = { paddingLeft: `${depth * 16 + 8}px` };

  return (
    <div>
      <div
        className="flex items-center gap-1.5 py-1 px-2 rounded text-sm transition-all cursor-pointer select-none hover:bg-gray-800 text-gray-300"
        style={indentStyle}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleRightClick}
        title={`${type} • ${displayName} • Double-click to open • Right-click to pin • ${entityId}`}
      >
        {/* Expand/Collapse Arrow */}
        {hasChildren && (
          <div 
            className="flex-shrink-0 w-4 h-4 flex items-center justify-center text-xs text-gray-500 hover:text-gray-300 cursor-pointer"
            onClick={handleExpandClick}
            title="Click to expand/collapse"
          >
            {isExpanded ? "▼" : "▶"}
          </div>
        )}
        
        {/* Empty space for items without children */}
        {!hasChildren && <div className="w-4 h-4 flex-shrink-0"></div>}
        
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
      
      {/* Render children when expanded */}
      {isExpanded && children.length > 0 && (
        <div className="mt-0.5 space-y-0.5">
          {children.map((child) => (
            <EntityItem
              key={child.data.entity_id}
              entityId={child.data.entity_id}
              onDoubleClick={onDoubleClick}
              onPin={onPin}
              expandedEntities={expandedEntities}
              onToggleExpanded={onToggleExpanded}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const EntityExplorer: React.FC<EntityExplorerProps> = ({
  rootEntityId,
  onEntityDoubleClick,
  onEntityPin,
  sendStrategyRequest,
  viewEntityId,
}) => {
  const [expandedEntities, setExpandedEntities] = useState<Set<string>>(new Set([rootEntityId]));
  const [searchQuery, setSearchQuery] = useState<string>('');
  const rootNode: any = useRecoilValue(nodeSelectorFamily(rootEntityId));
  const viewEntity: any = useRecoilValue(nodeSelectorFamily(viewEntityId || rootEntityId));

  const handleToggleExpanded = useCallback((entityId: string) => {
    setExpandedEntities(prev => {
      const newSet = new Set(prev);
      if (newSet.has(entityId)) {
        newSet.delete(entityId);
      } else {
        newSet.add(entityId);
      }
      return newSet;
    });
  }, []);

  // Search functionality
  const searchTargetEntityId = viewEntityId || rootEntityId;

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
  }, [searchQuery, searchTargetEntityId, viewEntityId, sendStrategyRequest]);

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

  if (!rootNode || !rootNode.data) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm">
        <div className="mb-2">🚫</div>
        <div>No root entity found</div>
        <div className="text-xs text-gray-600 mt-1">
          Entity ID: {rootEntityId}
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-lg">🌳</span>
          <h2 className="text-sm font-semibold text-white">Entity Explorer</h2>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Double-click entities to open them in new windows
        </div>
      </div>

      {/* Search Box */}
      {viewEntityId && (
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
      )}

      {/* Search Results */}
      {searchQuery.trim() && searchResults.length > 0 && (
        <div className="flex-shrink-0 border-b border-gray-800">
          <div className="p-2 bg-gray-800">
            <div className="text-xs text-gray-400 mb-2">
              🔍 Search Results ({searchResults.length})
            </div>
            <div className="max-h-48 overflow-auto space-y-1">
              {searchResults.map((entityId) => (
                <EntityItem
                  key={`search-${entityId}`}
                  entityId={entityId}
                  onDoubleClick={onEntityDoubleClick}
                  onPin={onEntityPin}
                  expandedEntities={new Set()}
                  onToggleExpanded={() => {}}
                  depth={0}
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

      {/* Entity Tree */}
      <div className="flex-1 overflow-auto p-2">
        <div className="text-xs text-gray-400 mb-2">
          📂 Entity Tree
        </div>
        <EntityItem
          entityId={rootEntityId}
          onDoubleClick={onEntityDoubleClick}
          onPin={onEntityPin}
          expandedEntities={expandedEntities}
          onToggleExpanded={handleToggleExpanded}
          depth={0}
        />
      </div>

      {/* Footer with Instructions */}
      <div className="flex-shrink-0 p-3 border-t border-gray-800 text-xs text-gray-500">
        <div className="space-y-1">
          <div>💡 <strong>Double-click</strong> to open entities</div>
          <div>📂 <strong>Single-click arrows</strong> to expand/collapse</div>
          <div>📌 <strong>Right-click</strong> to pin entities to dock</div>
          <div>🪟 Each entity opens in a separate window</div>
        </div>
      </div>
    </div>
  );
};
