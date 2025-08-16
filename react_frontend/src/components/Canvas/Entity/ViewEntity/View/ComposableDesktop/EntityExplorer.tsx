import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily, allEntitiesSelector, childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../EntityEnum';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { EntitySearchBox } from './EntitySearchBox';

// Standard view component props interface
interface EntityExplorerProps {
  data?: any;
  sendStrategyRequest: (request: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
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
  onDelete?: (entityId: string) => void;
  expandedEntities: Set<string>;
  onToggleExpanded: (entityId: string) => void;
  depth?: number;
}

const EntityItem: React.FC<EntityItemProps> = ({
  entityId,
  onDoubleClick,
  onPin,
  onDelete,
  expandedEntities,
  onToggleExpanded,
  depth = 0,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  const [showContextMenu, setShowContextMenu] = useState(false);

  // Close context menu when clicking elsewhere
  useEffect(() => {
    if (showContextMenu) {
      const handleGlobalClick = () => setShowContextMenu(false);
      document.addEventListener('click', handleGlobalClick);
      return () => document.removeEventListener('click', handleGlobalClick);
    }
  }, [showContextMenu]);

  // Define all hooks before any early returns
  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    // Close context menu if open
    setShowContextMenu(false);
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
    setShowContextMenu(true);
  }, []);

  const handlePinClick = useCallback((e: React.MouseEvent) => {
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
  
  // Get display name - for view entities, use view_component_type if available
  let displayName = node?.data?.name || 
                   node?.data?.entity_name || 
                   node?.data?.display_name ||
                   node?.data?.title ||
                   node?.data?.attributes?.name ||
                   node?.data?.attribute_map?.name ||
                   node?.data?.meta_data?.name ||
                   'Unnamed';

  // For view entities, prefer the view_component_type as the display name
  if (type === 'view' && node?.data?.view_component_type) {
    displayName = node.data.view_component_type;
  }

  // Get children entities if expanded
  const children = hasChildren && isExpanded 
    ? allEntities.filter(entity => node.data.child_ids.includes(entity.data.entity_id))
    : [];

  const indentStyle = { paddingLeft: `${depth * 16 + 8}px` };

  // Simple context menu component
  const ContextMenu = () => {
    if (!showContextMenu) return null;

    return (
      <div className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-600 rounded shadow-lg py-1 z-10 min-w-32">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowContextMenu(false);
            if (onPin) onPin(entityId);
          }}
          className="w-full px-3 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 flex items-center gap-2"
        >
          📌 Pin to dock
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowContextMenu(false);
            if (onDelete) onDelete(entityId);
          }}
          className="w-full px-3 py-2 text-left text-sm text-red-400 hover:bg-gray-700 flex items-center gap-2"
        >
          🗑️ Delete entity
        </button>
      </div>
    );
  };

  return (
    <div className="relative">
      <div
        className="flex items-center gap-1.5 py-1 px-2 rounded text-sm transition-all cursor-pointer select-none hover:bg-gray-800 text-gray-300"
        style={indentStyle}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleRightClick}
        title={`${type} • ${displayName} • Double-click to open • Click pin to dock • Right-click for menu • ${entityId}`}
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
        
        {/* Pin Icon */}
        <button
          onClick={handlePinClick}
          className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-xs text-gray-500 hover:text-yellow-400 hover:bg-gray-700 rounded transition-all"
          title="Pin to dock"
        >
          📌
        </button>
        
        {/* Entity Type Badge */}
        <div className="ml-auto text-[10px] text-gray-500 bg-gray-800 px-1 rounded">
          {type}
        </div>
      </div>
      
      {/* Context Menu */}
      <ContextMenu />
      
      {/* Render children when expanded */}
      {isExpanded && children.length > 0 && (
        <div className="mt-0.5 space-y-0.5">
          {children.map((child) => (
            <EntityItem
              key={child.data.entity_id}
              entityId={child.data.entity_id}
              onDoubleClick={onDoubleClick}
              onPin={onPin}
              onDelete={onDelete}
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
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}) => {
  // Extract configuration from data or use defaults
  const rootEntityId = data?.rootEntityId || parentEntityId;
  const onEntityDoubleClick = data?.onEntityDoubleClick || ((entityId: string) => {
    console.log('EntityExplorer: Entity double-clicked:', entityId);
  });
  const onEntityPin = data?.onEntityPin || ((entityId: string) => {
    // Default pinning behavior - store in view entity
    const request = StrategyRequests.setAttributes(viewEntityId, {
      pinned_entities: [...(data?.pinned_entities || []), entityId]
    });
    sendStrategyRequest(request);
  });
  
  const onEntityDelete = data?.onEntityDelete || ((entityId: string) => {
    setDeleteModalEntity(entityId);
  });
  const [expandedEntities, setExpandedEntities] = useState<Set<string>>(new Set([rootEntityId]));
  const [deleteModalEntity, setDeleteModalEntity] = useState<string | null>(null);
  const rootNode: any = useRecoilValue(nodeSelectorFamily(rootEntityId));
  const viewEntity: any = useRecoilValue(nodeSelectorFamily(viewEntityId || rootEntityId));
  
  // Get entity data for delete modal (hook must be called at top level, before any early returns)
  const deleteModalEntityNode: any = useRecoilValue(nodeSelectorFamily(deleteModalEntity || rootEntityId));

  // Get child view entities and check for EntitySearchBox view
  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: viewEntityId, type: EntityTypes.VIEW })) as any[];
  const entitySearchBoxView = useMemo(() => 
    viewChildren.find((view) => view.data?.view_component_type === 'entitysearchbox'),
    [viewChildren]
  );

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

  // Create EntitySearchBox view if it doesn't exist
  useEffect(() => {
    if (!entitySearchBoxView) {
      const entitySearchBoxRequest = StrategyRequests.builder()
        .withStrategyName('CreateEntityStrategy')
        .withTargetEntity(viewEntityId)
        .withParams({
          entity_class: 'shared_utils.entities.view_entity.ViewEntity.ViewEntity',
          initial_attributes: {
            parent_attributes: {},
            view_component_type: 'entitysearchbox',
            hidden: false
          }
        })
        .build();
      
      sendStrategyRequest(entitySearchBoxRequest);
    }
  }, [viewEntityId, entitySearchBoxView, sendStrategyRequest]);

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

  // Delete confirmation modal component
  const DeleteConfirmationModal = () => {
    if (!deleteModalEntity) return null;
    
    const entityName = deleteModalEntityNode?.data?.name || 
                      deleteModalEntityNode?.data?.entity_name || 
                      deleteModalEntityNode?.data?.display_name ||
                      deleteModalEntityNode?.data?.title ||
                      'this entity';

    const handleConfirmDelete = () => {
      // Send delete entity strategy request
      const deleteRequest = StrategyRequests.builder()
        .withStrategyName('RemoveEntityStrategy')
        .withTargetEntity(deleteModalEntity)
        .withParams({})
        .build();
      
      sendStrategyRequest(deleteRequest);
      setDeleteModalEntity(null);
    };

    const handleCancelDelete = () => {
      setDeleteModalEntity(null);
    };

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
        <div className="bg-gray-800 border border-gray-600 rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center gap-3 mb-4">
            <div className="text-2xl">⚠️</div>
            <div>
              <h3 className="text-lg font-semibold text-white">Confirm Delete</h3>
              <p className="text-sm text-gray-400">This action cannot be undone</p>
            </div>
          </div>
          
          <p className="text-gray-300 mb-6">
            Are you sure you want to delete <span className="font-medium text-white">"{entityName}"</span>? 
            This will permanently remove the entity and all its data.
          </p>
          
          <div className="flex gap-3 justify-end">
            <button
              onClick={handleCancelDelete}
              className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-all"
            >
              Cancel
            </button>
            <button
              onClick={handleConfirmDelete}
              className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 text-white rounded transition-all"
            >
              Delete Entity
            </button>
          </div>
        </div>
      </div>
    );
  };

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

      {/* Entity Search Box */}
      {viewEntityId && entitySearchBoxView && (
        <EntitySearchBox
          data={{
            onEntityDoubleClick: onEntityDoubleClick,
            onEntityPin: onEntityPin
          }}
          sendStrategyRequest={sendStrategyRequest}
          updateEntity={updateEntity}
          viewEntityId={entitySearchBoxView.data.entity_id}
          parentEntityId={viewEntityId}
        />
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
          onDelete={onEntityDelete}
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
          <div>📌 <strong>Click pin icon</strong> to pin entities to dock</div>
          <div>🪟 Each entity opens in a separate window</div>
        </div>
      </div>
      
      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal />
    </div>
  );
};