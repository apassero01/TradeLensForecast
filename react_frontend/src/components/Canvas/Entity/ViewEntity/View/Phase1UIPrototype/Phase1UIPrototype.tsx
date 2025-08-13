import React, { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { useRecoilValue, useRecoilCallback, useRecoilState } from 'recoil';
import { nodeSelectorFamily, childrenByTypeSelector, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import EntityViewRenderer from '../ChatInterface/EntityViewRenderer';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import { activeAgentIdAtom } from '../../../../../../state/activeAgentAtom';
import ActiveAgentIndicator from '../../../../../common/ActiveAgentIndicator';
import { useEntityNavigation } from '../../../../../../hooks/useEntityNavigation';
type Phase1UIPrototypeProps = {
  data?: any;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
};

// (Removed global render stack)

// Simple view component name mapping to avoid circular dependency
const VIEW_NAMES: Record<string, string> = {
  'chatinterface': 'Chat Interface',
  'advanced_document_editor': 'Advanced Document Editor',
  'histogram': 'Histogram',
  'linegraph': 'Line Graph',
  'editor': 'Editor',
  'entityrenderer': 'Entity Renderer',
  'phase1_ui_prototype': 'Phase 1 UI Prototype',
  'main_ui_phase1': 'Main UI (Phase 1)',
  // Add more as needed
};

// Better icons for common entity types
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

// Context Menu Component
function ContextMenu({ 
  x, 
  y, 
  entityId, 
  onMakeGlobalView, 
  onGoToActiveAgent, 
  onClose 
}: {
  x: number;
  y: number;
  entityId: string;
  onMakeGlobalView: (entityId: string) => void;
  onGoToActiveAgent: () => void;
  onClose: () => void;
}) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [onClose]);

  return (
    <div
      ref={menuRef}
      className="fixed bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 z-50 min-w-48"
      style={{ left: x, top: y }}
    >
      <button
        className="w-full px-3 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
        onClick={() => {
          onMakeGlobalView(entityId);
          onClose();
        }}
      >
        <span>🌍</span>
        <span>Make Global View</span>
      </button>
      <button
        className="w-full px-3 py-2 text-left text-sm text-gray-300 hover:bg-gray-700 hover:text-white flex items-center gap-2"
        onClick={() => {
          onGoToActiveAgent();
          onClose();
        }}
      >
        <span>🤖</span>
        <span>Go to Active Agent</span>
      </button>
    </div>
  );
}

// Active Agent Toggle Component
function ActiveAgentToggle({ 
  entityId, 
  isActive, 
  onSetActive 
}: {
  entityId: string;
  isActive: boolean;
  onSetActive: () => void;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onSetActive();
      }}
      className={`ml-2 px-2 py-0.5 text-[9px] font-medium rounded transition-all ${
        isActive 
          ? 'bg-green-600 text-white' 
          : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
      }`}
      title={isActive ? 'Active Agent' : 'Click to set as active agent'}
    >
      {isActive ? 'ACTIVE' : 'INACTIVE'}
    </button>
  );
}

// (Removed Agent View Creation Helper Component)

// Safe EntityViewRenderer that prevents circular dependencies
function SafeEntityViewRenderer({ 
  entityId, 
  initialViewId, 
  sendStrategyRequest, 
  updateEntity
}: {
  entityId: string;
  initialViewId?: string;
  sendStrategyRequest?: (request: any) => void;
  updateEntity?: (entityId: string, data: any) => void;
}) {
  return (
    <EntityViewRenderer
      entityId={entityId}
      initialViewId={initialViewId}
      sendStrategyRequest={sendStrategyRequest}
      updateEntity={updateEntity}
    />
  );
}

// Compact Entity Item Component
function EntityItem({ 
  entityId, 
  onSelect, 
  onNavigate,
  onReparent, 
  onViewSelect,
  onToggleExpanded,
  onContextMenu,
  expandedEntities,
  activeAgentId,
  setActiveAgentId,
  isCurrentView = false, 
  isParent = false,
  isExpanded = false,
  showAsGroup = false,
  groupType = '',
  depth = 0,
}: {
  entityId: string;
  onSelect: (id: string) => void;
  onNavigate?: (id: string) => void;
  onReparent: (childId: string, newParentId: string) => void;
  onViewSelect?: (viewId: string) => void;
  onToggleExpanded?: (id: string) => void;
  onContextMenu?: (x: number, y: number, entityId: string) => void;
  expandedEntities: Set<string>;
  activeAgentId: unknown;
  setActiveAgentId: (id: unknown) => void;
  isCurrentView?: boolean;
  isParent?: boolean;
  isExpanded?: boolean;
  showAsGroup?: boolean;
  groupType?: string;
  depth?: number;
}) {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  const draggableRef = useRef<HTMLDivElement>(null);

  if (!node || !node.data) {
    return (
      <div className="px-2 py-0.5 text-xs text-gray-600">(missing)</div>
    );
  }

  const type = node?.data?.entity_type || 'entity';
  const hasChildren = node?.data?.child_ids && node?.data?.child_ids.length > 0;
  // Fix agent naming - check name attribute first, then other fallbacks
  let displayName = node?.data?.name || 
                   node?.data?.entity_name || 
                   node?.data?.display_name ||
                   node?.data?.title ||
                   node?.data?.attributes?.name ||
                   node?.data?.attribute_map?.name ||
                   node?.data?.meta_data?.name ||
                   node?.data?.entity_type || 
                   'Unnamed';
  
  // Use view component display name for views
  if (type === EntityTypes.VIEW || type === 'view') {
    const vType = node?.data?.view_component_type;
    displayName = VIEW_NAMES[vType] || vType || displayName;
  }

  // Get children entities if expanded
  const children = hasChildren && isExpanded 
    ? allEntities.filter(entity => node.data.child_ids.includes(entity.data.entity_id))
    : [];

  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/x-entity-id', node.data.entity_id);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const childId = e.dataTransfer.getData('application/x-entity-id');
    if (!childId || childId === node.data.entity_id) return;
    onReparent(childId, node.data.entity_id);
  };

  const isViewEntity = type === EntityTypes.VIEW || type === 'view';
  
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (isParent && onNavigate) {
      // Parent nodes should navigate to that entity (change entire app context)
      onNavigate(node.data.entity_id);
    } else if (isViewEntity && onViewSelect && !isCurrentView) {
      // For child view entities, set them as the current view instead of navigating
      onViewSelect(node.data.entity_id);
    } else {
      // For other child entities, use normal selection (render in center)
      onSelect(node.data.entity_id);
    }
  };

  const handleExpandClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hasChildren && onToggleExpanded && !isParent) {
      onToggleExpanded(node.data.entity_id);
    }
  };

  const handleRightClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onContextMenu) {
      onContextMenu(e.clientX, e.clientY, node.data.entity_id);
    }
  };

  const baseClasses = "flex items-center gap-1.5 px-2 py-0.5 rounded text-xs transition-all cursor-pointer select-none";
  const stateClasses = isCurrentView 
    ? "bg-blue-700 border border-blue-500 text-white shadow-sm" 
    : isParent
    ? "hover:bg-gray-750 text-gray-300 border-l-2 border-gray-600"
    : isViewEntity
    ? "hover:bg-purple-800 text-purple-300 border-l-2 border-purple-600"
    : "hover:bg-gray-800 text-gray-400";

  return (
    <>
    <div
      ref={draggableRef}
      draggable
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={`${baseClasses} ${stateClasses}`}
      onClick={handleClick}
      onContextMenu={handleRightClick}
      title={`${type} • ${displayName} • ${
        hasChildren && !isParent 
          ? 'Click arrow to expand/collapse' 
          : isParent 
            ? 'Click to navigate to parent' 
            : isViewEntity && !isCurrentView 
              ? 'Click to set as current view' 
              : 'Click to navigate'
      } • Right-click for menu • ${node.data.entity_id}`}
    >
      {hasChildren && !isParent && (
        <div 
          className="flex-shrink-0 text-xs text-gray-500 hover:text-gray-300 cursor-pointer p-0.5"
          onClick={handleExpandClick}
          title="Click to expand/collapse"
        >
          {isExpanded ? "▼" : "▶"}
        </div>
      )}
      <div className="flex-shrink-0 text-sm">
        {getEntityIcon(type)}
      </div>
      <div className="truncate min-w-0 font-medium">
        {isParent && "↑ "}{displayName}
        {isCurrentView && " 📍"}
      </div>
      
      {/* Active Agent Indicator for API Models */}
      {type === 'api_model' && (
        <ActiveAgentToggle 
          entityId={node.data.entity_id}
          isActive={activeAgentId === node.data.entity_id}
          onSetActive={() => {
            if (activeAgentId !== node.data.entity_id) {
              setActiveAgentId(node.data.entity_id);
              // Also navigate to this agent (change file tree context)
              if (onNavigate) {
                onNavigate(node.data.entity_id);
              }
            }
          }}
        />
      )}
      
      {showAsGroup && (
        <div className="ml-auto text-[10px] text-gray-500 bg-gray-800 px-1 rounded">
          {groupType}
        </div>
      )}
    </div>
    
    {/* Render children when expanded */}
    {isExpanded && children.length > 0 && (
      <div className="ml-4 mt-1 space-y-0.5">
        {children.map((child) => (
          <EntityItem
            key={child.data.entity_id}
            entityId={child.data.entity_id}
            onSelect={onSelect}
            onNavigate={onNavigate}
            onReparent={onReparent}
            onViewSelect={onViewSelect}
            onToggleExpanded={onToggleExpanded}
            onContextMenu={onContextMenu}
            expandedEntities={expandedEntities}
            activeAgentId={activeAgentId}
            setActiveAgentId={setActiveAgentId}
            isExpanded={expandedEntities.has(child.data.entity_id)}
            depth={depth + 1}
          />
        ))}
      </div>
    )}
  </>
  );
}

// Enhanced Entity Explorer with Parent/Current/Children hierarchy
function EntityTreeExplorer({ 
  currentEntityId, 
  onSelect, 
  onNavigate,
  onReparent,
  onViewSelect,
  onToggleExpanded,
  onContextMenu,
  expandedEntities,
  activeAgentId,
  setActiveAgentId
}: {
  currentEntityId: string;
  onSelect: (id: string) => void;
  onNavigate: (id: string) => void;
  onReparent: (childId: string, newParentId: string) => void;
  onViewSelect?: (viewId: string) => void;
  onToggleExpanded?: (id: string) => void;
  onContextMenu?: (x: number, y: number, entityId: string) => void;
  expandedEntities: Set<string>;
  activeAgentId: unknown;
  setActiveAgentId: (id: unknown) => void;
}) {
  const currentNode: any = useRecoilValue(nodeSelectorFamily(currentEntityId));
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  
  if (!currentNode || !currentNode.data) {
    return (
      <div className="p-3 text-center text-gray-500 text-sm">
        No entity selected
      </div>
    );
  }

  // Get parent entity
  const parentId = currentNode.data.parent_ids?.[0];
  
  // Get children and group them by type
  const childIds: string[] = currentNode.data.child_ids || [];
  const children = childIds.map(id => allEntities.find(e => e.id === id)).filter(Boolean);
  
  // Group children by entity type
  const groupedChildren = children.reduce((groups: Record<string, any[]>, child) => {
    const type = child.data?.entity_type || 'entity';
    if (!groups[type]) groups[type] = [];
    groups[type].push(child);
    return groups;
  }, {});

  // Sort group types for consistent ordering - API models right after views
  const sortedGroupTypes = Object.keys(groupedChildren).sort((a, b) => {
    const priority = ['view', 'api_model', 'agent', 'document'];
    const aIndex = priority.indexOf(a);
    const bIndex = priority.indexOf(b);
    if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  return (
    <div className="space-y-1 p-2">
      {/* Parent Section */}
      {parentId && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1 px-2">Parent</div>
          <EntityItem
            entityId={parentId}
            onSelect={onSelect}
            onNavigate={onNavigate}
            onReparent={onReparent}
            onViewSelect={onViewSelect}
            onToggleExpanded={onToggleExpanded}
            onContextMenu={onContextMenu}
            expandedEntities={expandedEntities}
            activeAgentId={activeAgentId}
            setActiveAgentId={setActiveAgentId}
            isParent={true}
            isExpanded={expandedEntities.has(parentId)}
          />
        </div>
      )}

      {/* Current Entity */}
      <div className="mb-3">
        <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1 px-2">Current</div>
        <EntityItem
          entityId={currentEntityId}
          onSelect={onSelect}
          onNavigate={onNavigate}
          onReparent={onReparent}
          onViewSelect={onViewSelect}
          onToggleExpanded={onToggleExpanded}
          onContextMenu={onContextMenu}
          expandedEntities={expandedEntities}
          activeAgentId={activeAgentId}
          setActiveAgentId={setActiveAgentId}
          isCurrentView={true}
          isExpanded={expandedEntities.has(currentEntityId)}
        />
      </div>

      {/* Children Grouped by Type */}
      {sortedGroupTypes.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wide text-gray-500 mb-1 px-2">
            Children ({children.length})
          </div>
          <div className="space-y-2">
            {sortedGroupTypes.map((groupType) => (
              <div key={groupType} className="space-y-0.5">
                {groupedChildren[groupType].length > 1 && (
                  <div className="text-[9px] uppercase tracking-wide text-gray-600 px-2 flex items-center gap-1">
                    <span>{getEntityIcon(groupType)}</span>
                    <span>{groupType}s ({groupedChildren[groupType].length})</span>
                  </div>
                )}
                {groupedChildren[groupType].map((child) => (
                  <EntityItem
                    key={child.data.entity_id}
                    entityId={child.data.entity_id}
                    onSelect={onSelect}
                    onNavigate={onNavigate}
                    onReparent={onReparent}
                    onViewSelect={onViewSelect}
                    onToggleExpanded={onToggleExpanded}
                    onContextMenu={onContextMenu}
                    expandedEntities={expandedEntities}
                    activeAgentId={activeAgentId}
                    setActiveAgentId={setActiveAgentId}
                    isExpanded={expandedEntities.has(child.data.entity_id)}
                    showAsGroup={groupedChildren[groupType].length === 1}
                    groupType={groupType}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Children Message */}
      {children.length === 0 && (
        <div className="text-center text-gray-600 text-xs py-3">
          <div className="mb-1">📭</div>
          <div>No children entities</div>
        </div>
      )}
    </div>
  );
}

export default function Phase1UIPrototype({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: Phase1UIPrototypeProps) {
  
  // Get all entities for agent dashboard
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  
  // Use frontend state management for active agent
  const [activeAgentId, setActiveAgentId] = useRecoilState(activeAgentIdAtom);
  const activeAgentNode: any = useRecoilValue(nodeSelectorFamily(activeAgentId || ''));
  
  // Use proper top-level navigation
  const { navigateToEntity, navigateToCanvas, currentEntityId, isInEntityView } = useEntityNavigation();
  
  // The entity we should show context for is the current session entity (the one we navigated to)
  // If we're in entity view mode, use that entity; otherwise fallback to parentEntityId
  const contextEntityId = isInEntityView && currentEntityId ? currentEntityId : parentEntityId;
  
  // Safety check: prevent circular rendering if Phase1UIPrototype is used inappropriately
  const viewEntityNode = useRecoilValue(nodeSelectorFamily(viewEntityId || ''));
  const contextNode = useRecoilValue(nodeSelectorFamily(contextEntityId || ''));
  
  // (Removed circular dependency detection)
  
  // Sync with backend when component mounts (read from view entity if available)
  const viewEntityState: any = useRecoilValue(nodeSelectorFamily(viewEntityId || ''));
  useEffect(() => {
    const backendActiveAgent = viewEntityState?.data?.active_agent_id;
    if (backendActiveAgent && backendActiveAgent !== activeAgentId) {
      setActiveAgentId(backendActiveAgent);
    }
  }, [viewEntityState?.data?.active_agent_id, activeAgentId, setActiveAgentId]);
  

  
  // Content state
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [selectedInitialViewId, setSelectedInitialViewId] = useState<string | null>(null);
  const [showAgentDashboard, setShowAgentDashboard] = useState<boolean>(false);
  
  // View selection state for the current context entity
  const [currentEntityViewId, setCurrentEntityViewId] = useState<string | null>(null);
  
  // Entity expansion state for file tree
  const [expandedEntities, setExpandedEntities] = useState<Set<string>>(new Set());
  
  // Context menu state
  const [contextMenu, setContextMenu] = useState<{x: number, y: number, entityId: string} | null>(null);
  
  // (Removed agent view creation helper state)
  
  // Circular dependency prevention
  // (Removed circular dependency prevention state)

  // Global Chat input state
  const [globalInput, setGlobalInput] = useState('');
  const [sending, setSending] = useState(false);

  // Helper functions using useRecoilCallback to avoid stale closures
  const getNode = useRecoilCallback(({ snapshot }) => async (id: string) => {
    if (!id) return null as any;
    return await snapshot.getPromise(nodeSelectorFamily(id));
  }, []);

  const getViewChildOfType = useRecoilCallback(({ snapshot }) => async (parentId: string, viewType: string) => {
    if (!parentId) return null as any;
    const views = await snapshot.getPromise(childrenByTypeSelector({ parentId, type: EntityTypes.VIEW }));
    const match = (views as any[]).find(v => v?.data?.view_component_type === viewType);
    return match?.data?.entity_id || null;
  }, []);

  // Navigation functions for top-level entity changes
  const handleEntityNavigation = useCallback((entityId: string) => {
    if (entityId === contextEntityId) return;
    
    // Navigate to entity at the top level (changes entire app context)
    navigateToEntity(entityId);
  }, [contextEntityId, navigateToEntity]);

  const handleViewSelection = useCallback((viewId: string) => {
    // Set this view as the current view for the context entity
    setCurrentEntityViewId(viewId);
    
    // Clear other selections to show the main content area
    setSelectedEntityId(null);
    setSelectedInitialViewId(null);
    setShowAgentDashboard(false);
  }, []);

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

  const handleMakeGlobalView = useCallback((entityId: string) => {
    // Navigate to this entity as the global view
    handleEntityNavigation(entityId);
  }, [handleEntityNavigation]);

  const handleGoToActiveAgent = useCallback(() => {
    if (activeAgentId && typeof activeAgentId === 'string') {
      handleEntityNavigation(activeAgentId);
    }
  }, [activeAgentId, handleEntityNavigation]);

  const handleContextMenu = useCallback((x: number, y: number, entityId: string) => {
    setContextMenu({ x, y, entityId });
  }, []);

  const handleCloseContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  // (Removed agent view creation handlers)



  // (Removed blockView)



  // (Removed handleCircularDependency)

  const handleBackToCanvas = useCallback(() => {
    navigateToCanvas();
  }, [navigateToCanvas]);

  const handleNavigateToActiveAgent = useCallback(() => {
    if (activeAgentId && typeof activeAgentId === 'string' && activeAgentId !== contextEntityId) {
      navigateToEntity(activeAgentId);
    }
  }, [activeAgentId, contextEntityId, navigateToEntity]);

  // Entity selection handler - CRITICAL: Always show Agent Dashboard for api_models
  const handleSelectEntity = useCallback(async (entityId: string) => {
    // Prevent selecting the current view entity to avoid recursion
    if (entityId === viewEntityId) {
      console.warn('Prevented recursive view selection');
      return;
    }
    
    setShowAgentDashboard(false);
    setSelectedInitialViewId(null);
    
    const node = await getNode(entityId);
    const type = node?.data?.entity_type;
    
    // If clicking api_model - show chat interface (not agent dashboard)
    if (node?.data?.entity_name === 'api_model') {
      // For api_model entities, show them in the center with a chat-focused view
      setSelectedEntityId(entityId);
      // Try to find a chat view for this agent
      const chatViewId = await getViewChildOfType(entityId, 'entity_centric_chat_view');
      if (chatViewId) {
        setSelectedInitialViewId(chatViewId);
      }
      return;
    }
    
    // Document: open editor view
    if (type === EntityTypes.DOCUMENT || type === 'document') {
      const editorViewId = await getViewChildOfType(entityId, 'advanced_document_editor');
      setSelectedEntityId(entityId);
      if (editorViewId) setSelectedInitialViewId(editorViewId);
      return;
    }
    
    // Default: just select entity
    setSelectedEntityId(entityId);
  }, [getNode, getViewChildOfType, viewEntityId]);

  // Reparent handler
  const handleReparentAsync = useCallback(async (childId: string, newParentId: string) => {
    if (!childId || !newParentId || childId === newParentId) return;
    
    const childNode: any = await getNode(childId);
    const oldParentId = childNode?.data?.parent_ids?.[0];

    const requests: any[] = [];
    requests.push(
      StrategyRequests.builder()
        .withStrategyName('AddChildStrategy')
        .withTargetEntity(newParentId)
        .withParams({ child_id: childId })
        .withAddToHistory(false)
        .build()
    );
    
    if (oldParentId && oldParentId !== newParentId) {
      requests.push(
        StrategyRequests.builder()
          .withStrategyName('RemoveChildStrategy')
          .withTargetEntity(oldParentId)
          .withParams({ child_id: childId })
          .withAddToHistory(false)
          .build()
      );
    }
    
    sendStrategyRequest(requests);
  }, [getNode, sendStrategyRequest]);

  // Smart Context management
  const pinnedIds: string[] = useMemo(() => 
    Array.isArray(activeAgentNode?.data?.pinned_entity_ids) ? activeAgentNode?.data?.pinned_entity_ids : [], 
    [activeAgentNode?.data?.pinned_entity_ids]
  );
  
  const workingIds: string[] = useMemo(() => {
    const visibles = Array.isArray(activeAgentNode?.data?.visible_entities) ? activeAgentNode?.data?.visible_entities : [];
    return visibles.filter((id: string) => !pinnedIds.includes(id));
  }, [activeAgentNode?.data?.visible_entities, pinnedIds]);

  const pinEntity = useCallback((entityId: string) => {
    if (!activeAgentNode) return;
    const nextPinned = Array.from(new Set([...(pinnedIds || []), entityId]));
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(activeAgentNode.entity_id)
        .withParams({ attribute_map: { pinned_entity_ids: nextPinned } })
        .withAddToHistory(false)
        .build()
    );
  }, [activeAgentNode, pinnedIds, sendStrategyRequest]);

  const removeFromWorking = useCallback((entityId: string) => {
    if (!activeAgentNode) return;
    const visibles: string[] = activeAgentNode?.data?.visible_entities || [];
    const next = visibles.filter((id) => id !== entityId);
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(activeAgentNode.entity_id)
        .withParams({ attribute_map: { visible_entities: next } })
        .withAddToHistory(false)
        .build()
    );
  }, [activeAgentNode, sendStrategyRequest]);

  // Drag handlers for pinning
  const handlePinnedDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  
  const handlePinnedDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const draggedId = e.dataTransfer.getData('application/x-entity-id');
    if (draggedId) pinEntity(draggedId);
  };

  // Global chat functions
  const sendGlobalMessage = async () => {
    if (!activeAgentNode || !globalInput.trim() || sending) return;
    setSending(true);
    try {
      sendStrategyRequest(
        StrategyRequests.builder()
          .withStrategyName('CallApiModelStrategy')
          .withTargetEntity(activeAgentNode.entity_id)
          .withParams({ user_input: globalInput, serialize_entities_and_strategies: true })
          .withAddToHistory(false)
          .build()
      );
      setGlobalInput('');
    } finally {
      setSending(false);
    }
  };

  const handleGlobalInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendGlobalMessage();
    }
  };

  // Agent Dashboard Component - Primary interface for managing agents
  const AgentDashboard: React.FC = () => {
    const agentEntities = allEntities.filter((e: any) => e?.data?.entity_name === 'api_model');
    const [editingAgent, setEditingAgent] = useState<string | null>(null);
    const [editingName, setEditingName] = useState<string>('');

    const startEditing = (agentId: string, currentName: string) => {
      setEditingAgent(agentId);
      setEditingName(currentName);
    };

    const cancelEditing = () => {
      setEditingAgent(null);
      setEditingName('');
    };

    const saveAgentName = (agentId: string) => {
      if (editingName.trim()) {
        sendStrategyRequest(
          StrategyRequests.builder()
            .withStrategyName('SetAttributesStrategy')
            .withTargetEntity(agentId)
            .withParams({ attribute_map: { name: editingName.trim() } })
            .withAddToHistory(false)
            .build()
        );
      }
      cancelEditing();
    };

    const handleNameKeyDown = (e: React.KeyboardEvent, agentId: string) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        saveAgentName(agentId);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        cancelEditing();
      }
    };
    
    if (agentEntities.length === 0) {
      return (
        <div className="h-full grid place-items-center text-gray-500">
          <div className="text-center">
            <div className="text-lg mb-2">🤖 No AI Agents Found</div>
            <div className="text-sm">Create an API model entity to get started with AI assistance.</div>
          </div>
        </div>
      );
    }

    return (
      <div className="h-full flex flex-col">
        <div className="flex-shrink-0 p-6 pb-4">
          <h2 className="text-xl font-bold text-white mb-2">Agent Dashboard</h2>
          <p className="text-gray-400">Select and manage your AI agents. The active agent will handle your messages and context.</p>
        </div>
        
        <div className="flex-1 overflow-auto px-6 pb-6">
          <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
            {agentEntities.map((agent: any) => {
              const isActive = activeAgentId === agent.data.entity_id;

              // Try multiple possible attribute names for agent name
              const agentName = agent.data.name || 
                               agent.data.entity_name || 
                               agent.data.display_name ||
                               agent.data.title ||
                               agent.data.attributes?.name ||
                               agent.data.attribute_map?.name ||
                               agent.data.meta_data?.name ||
                               `Agent ${agent.data.entity_id.slice(0, 8)}`;
              const isEditing = editingAgent === agent.data.entity_id;
              
              return (
                <div 
                  key={agent.data.entity_id} 
                  className={`relative bg-gray-800 border rounded-lg p-4 flex flex-col gap-3 transition-all hover:bg-gray-750 ${
                    isActive ? 'border-blue-500 ring-2 ring-blue-500/20' : 'border-gray-700'
                  }`}
                >
                  {isActive && (
                    <div className="absolute -top-2 -right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full font-semibold">
                      ACTIVE
                    </div>
                  )}
                  
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 bg-blue-600 rounded-lg grid place-items-center text-white font-bold">
                        🤖
                      </div>
                      <div className="text-sm text-gray-400">API Model</div>
                    </div>
                    <button
                      className="text-gray-400 hover:text-white text-sm p-1 rounded transition-colors"
                      onClick={() => startEditing(agent.data.entity_id, agentName)}
                      title="Edit agent name"
                      disabled={isEditing}
                    >
                      ✏️
                    </button>
                  </div>
                  
                  {isEditing ? (
                    <div className="space-y-2">
                      <input
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onKeyDown={(e) => handleNameKeyDown(e, agent.data.entity_id)}
                        onBlur={() => saveAgentName(agent.data.entity_id)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="Enter agent name..."
                        autoFocus
                        maxLength={50}
                      />
                      <div className="flex gap-2">
                        <button
                          className="flex-1 px-2 py-1 text-xs bg-blue-600 hover:bg-blue-500 rounded transition-colors"
                          onClick={() => saveAgentName(agent.data.entity_id)}
                        >
                          ✅ Save
                        </button>
                        <button
                          className="flex-1 px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded transition-colors"
                          onClick={cancelEditing}
                        >
                          ❌ Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div 
                      className="text-lg font-semibold text-white truncate cursor-pointer hover:text-blue-300 transition-colors" 
                      title={`${agentName} (click edit icon to rename)`}
                      onClick={() => startEditing(agent.data.entity_id, agentName)}
                    >
                      {agentName}
                    </div>
                  )}
                  
                  <div className="text-xs text-gray-500 font-mono break-all">
                    {agent.data.entity_id}
                  </div>
                  
                  <div className="flex gap-4 text-xs text-gray-400 mt-2">
                    <div>
                      <span className="text-gray-500">📌 Pinned:</span> {(agent.data.pinned_entity_ids || []).length}
                    </div>
                    <div>
                      <span className="text-gray-500">🔄 Working:</span> {(agent.data.visible_entities || []).length}
                    </div>
                  </div>
                  
                  <div className="flex gap-2 mt-auto">
                    <button
                      className={`flex-1 px-3 py-2 rounded text-sm font-medium transition-colors ${
                        isActive 
                          ? 'bg-blue-600 text-white cursor-default' 
                          : 'bg-blue-600 hover:bg-blue-500 text-white'
                      }`}
                                          onClick={() => {
                      if (!isActive) {
                        // Update frontend state immediately for instant UI feedback
                        setActiveAgentId(agent.data.entity_id);
                        
                        // Navigate to this agent's entity view
                        handleEntityNavigation(agent.data.entity_id);
                        
                        // Also update backend for persistence (optional)
                        sendStrategyRequest(
                          StrategyRequests.builder()
                            .withStrategyName('SetAttributesStrategy')
                            .withTargetEntity(viewEntityId)
                            .withParams({ attribute_map: { active_agent_id: agent.data.entity_id } })
                            .withAddToHistory(false)
                            .build()
                        );
                      }
                    }}
                      disabled={isActive || isEditing}
                    >
                      {isActive ? '✅ Active Agent' : 'Set as Active'}
                    </button>
                    
                    <button
                      className="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-sm font-medium text-white transition-colors disabled:opacity-50"
                                          onClick={() => {
                      // Navigate to this agent's entity view (will show chat interface)
                      handleEntityNavigation(agent.data.entity_id);
                    }}
                      title="Open dedicated chat interface for this agent"
                      disabled={isEditing}
                    >
                      💬 Chat
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>
    );
  };

  // Current View Breadcrumb component
  const CurrentViewBreadcrumb: React.FC<{ entityId: string }> = ({ entityId }) => {
    const node: any = useRecoilValue(nodeSelectorFamily(entityId || ''));
    if (!node || !node.data) return <span className="text-gray-500">Unknown</span>;
    
    const type = node?.data?.entity_type || 'entity';
    const name = node?.data?.name || node?.data?.entity_name || type;
    
    return (
      <div className="flex items-center gap-2">
        <div className="text-sm">
          {getEntityIcon(type)}
        </div>
        <span className="text-white text-sm font-medium">{name}</span>
      </div>
    );
  };

  // Entity Pill component for context display - now clickable for navigation
  const EntityPill: React.FC<{ id: string; onClose?: () => void; draggable?: boolean; showNavigation?: boolean }> = ({ id, onClose, draggable, showNavigation = false }) => {
    const n: any = useRecoilValue(nodeSelectorFamily(id || ''));
    if (!id || !n || !n.data) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm text-gray-500">
          (missing)
        </div>
      );
    }
    
    const t = n?.data?.entity_type || 'entity';
    const name = n?.data?.name || 
                n?.data?.entity_name || 
                n?.data?.display_name ||
                n?.data?.title ||
                n?.data?.attributes?.name ||
                n?.data?.attribute_map?.name ||
                n?.data?.meta_data?.name ||
                t;
    const isCurrentView = id === contextEntityId;
    
    const handleDragStart = (e: React.DragEvent) => {
      if (!draggable) return;
      e.dataTransfer.setData('application/x-entity-id', id);
    };

    const handleClick = (e: React.MouseEvent) => {
      // Don't navigate if clicking action buttons
      if ((e.target as HTMLElement).closest('button[aria-label="remove"]') || 
          (e.target as HTMLElement).closest('button[aria-label="navigate"]') ||
          (e.target as HTMLElement).closest('button[aria-label="view"]')) return;
      
      // Default behavior: render view in center
      handleSelectEntity(id);
    };

    const handleNavigateClick = (e: React.MouseEvent) => {
      e.stopPropagation();
      // Navigate to entity (change file tree context)
      handleEntityNavigation(id);
    };

    const handleViewClick = (e: React.MouseEvent) => {
      e.stopPropagation();
      // Render view in center
      handleSelectEntity(id);
    };
    
    return (
      <div
        className={`flex items-center gap-2 px-2 py-1 rounded border text-sm transition-all cursor-pointer hover:bg-gray-700 ${
          isCurrentView 
            ? 'bg-blue-800 border-blue-600 ring-1 ring-blue-500' 
            : 'bg-gray-800 border-gray-700'
        }`}
        draggable={draggable}
        onDragStart={handleDragStart}
        onClick={handleClick}
        title={`${t} • ${id} • Click to navigate`}
      >
        <div className="flex-shrink-0 text-sm">
          {getEntityIcon(t)}
        </div>
        <span className="truncate max-w-[100px] text-gray-200">{name}</span>
        {isCurrentView && <span className="text-blue-400 text-xs">📍</span>}
        
        {/* Navigation Arrows */}
        {showNavigation && (
          <div className="flex items-center gap-1 ml-auto">
            <button
              className="p-1 text-gray-400 hover:text-blue-400 hover:bg-gray-700 rounded transition-all"
              onClick={handleNavigateClick}
              aria-label="navigate"
              title="Navigate to entity (change file tree context)"
            >
              ↗️
            </button>
            <button
              className="p-1 text-gray-400 hover:text-green-400 hover:bg-gray-700 rounded transition-all"
              onClick={handleViewClick}
              aria-label="view"
              title="Show entity view in center"
            >
              👁️
            </button>
          </div>
        )}
        
        {onClose && (
          <button 
            className="text-gray-400 hover:text-white ml-auto" 
            onClick={(e) => { e.stopPropagation(); onClose(); }} 
            aria-label="remove"
          >
            ×
          </button>
        )}
      </div>
    );
  };

  // (Removed circular dependency fallback)

  return (
    <div className="nodrag flex flex-col w-full h-full bg-gray-900 text-white">
      {/* Header with Navigation */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800 bg-gray-900/95 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-white">Phase 1 UI</h1>
            
            {/* Navigation Breadcrumbs */}
            <div className="flex items-center gap-2 text-sm">
              {isInEntityView && (
                <button
                  onClick={handleBackToCanvas}
                  className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs transition-colors"
                  title="Back to Canvas"
                >
                  ← Canvas
                </button>
              )}
              
              {/* Current Context Entity */}
              {contextEntityId && (
                <div className="flex items-center gap-2 px-2 py-1 rounded bg-gray-800 border border-gray-700">
                  <div className="text-gray-400 text-xs">Context:</div>
                  <CurrentViewBreadcrumb entityId={contextEntityId} />
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Active Agent Button - navigate to active agent */}
            {activeAgentId && activeAgentId !== contextEntityId && (
              <button
                className="px-3 py-1 text-xs rounded bg-blue-700 hover:bg-blue-600 border border-blue-600 transition-colors"
                onClick={handleNavigateToActiveAgent}
                title="Navigate to active agent"
              >
                🤖 Go to Active Agent
              </button>
            )}
            
            <button
              className="px-3 py-1 text-xs rounded bg-gray-800 hover:bg-gray-700 border border-gray-700"
              onClick={() => setShowAgentDashboard(!showAgentDashboard)}
              title="Agent Management"
            >
              🤖 Agents
            </button>
            
            <ActiveAgentIndicator />
          </div>
        </div>
      </div>

      {/* Main body: Left • Center • Right */}
      <div className="flex-1 min-h-0 flex">
        {/* Left Sidebar: Enhanced Entity Explorer */}
        <div className="w-80 border-r border-gray-800 flex flex-col">
          <div className="p-3 text-xs text-gray-400 border-b border-gray-800 flex items-center gap-2">
            <span>🌳</span>
            <span>Entity Explorer</span>
          </div>
          <div className="flex-1 overflow-auto nowheel">
            {contextEntityId && (
              <EntityTreeExplorer
                currentEntityId={contextEntityId}
                onSelect={handleSelectEntity}
                onNavigate={handleEntityNavigation}
                onReparent={handleReparentAsync}
                onViewSelect={handleViewSelection}
                onToggleExpanded={handleToggleExpanded}
                onContextMenu={handleContextMenu}
                expandedEntities={expandedEntities}
                activeAgentId={activeAgentId}
                setActiveAgentId={(id: string) => setActiveAgentId(id)}
              />
            )}
          </div>
        </div>

        {/* Center: Main Content Area */}
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex-shrink-0 p-2 border-b border-gray-800 text-xs text-gray-400 flex items-center justify-between">
            <span>
              {showAgentDashboard ? '🤖 Agent Dashboard' : selectedEntityId ? `📄 Selected: ${selectedEntityId.slice(0, 8)}...` : '👋 Welcome'}
            </span>
            {/* (Removed blocked views controls) */}
          </div>
          <div className="flex-1 overflow-hidden">
            {showAgentDashboard ? (
              <AgentDashboard />
            ) : selectedEntityId ? (
              <SafeEntityViewRenderer
                key={`entity-${selectedEntityId}-${selectedInitialViewId || 'default'}`}
                entityId={selectedEntityId}
                initialViewId={selectedInitialViewId || undefined}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity}
              />
            ) : (
              // Show the current context entity with selected view if available
              <SafeEntityViewRenderer
                key={`current-context-${contextEntityId}-${currentEntityViewId || 'default'}`}
                entityId={contextEntityId}
                initialViewId={currentEntityViewId || undefined}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity}
              />
            )}
          </div>
        </div>

        {/* Right Sidebar: Smart Context */}
        <div className="w-80 border-l border-gray-800 flex flex-col">
          <div className="p-3 border-b border-gray-800 text-xs text-gray-400">🧠 Smart Context</div>
          <div className="p-3 space-y-4 overflow-auto nowheel">
            {/* Pinned Context */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs uppercase tracking-wide text-gray-400">📌 Pinned Context</div>
              </div>
              <div
                className="min-h-16 p-3 rounded bg-gray-900 border border-gray-800 border-dashed flex flex-wrap gap-2 transition-colors hover:border-gray-700"
                onDragOver={handlePinnedDragOver}
                onDrop={handlePinnedDrop}
                title="Drag entities from Working Context here to pin them permanently"
              >
                {pinnedIds.length === 0 ? (
                  <div className="w-full text-center text-xs text-gray-600 py-2">
                    <div className="mb-1">📌 No pinned entities</div>
                    <div className="text-gray-700">Drag items here to keep them always visible to your agent</div>
                  </div>
                ) : (
                  pinnedIds.map((id) => (
                    <EntityPill key={id} id={id} showNavigation />
                  ))
                )}
              </div>
            </div>

            {/* Working Context */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs uppercase tracking-wide text-gray-400">🔄 Working Context</div>
              </div>
              <div className="min-h-16 p-3 rounded bg-gray-900 border border-gray-800 flex flex-wrap gap-2">
                {workingIds.length === 0 ? (
                  <div className="w-full text-center text-xs text-gray-600 py-2">
                    <div className="mb-1">🔄 No working context</div>
                    <div className="text-gray-700">
                      {activeAgentNode 
                        ? "Your agent hasn't added any entities to its working context yet"
                        : "Select an active agent to see its working context"
                      }
                    </div>
                  </div>
                ) : (
                  workingIds.map((id) => (
                    <EntityPill key={id} id={id} draggable showNavigation onClose={() => removeFromWorking(id)} />
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          entityId={contextMenu.entityId}
          onMakeGlobalView={handleMakeGlobalView}
          onGoToActiveAgent={handleGoToActiveAgent}
          onClose={handleCloseContextMenu}
        />
      )}

      {/* Bottom: Global Chat Input */}
      <div className="flex-shrink-0 border-t border-gray-800 p-3 bg-gray-900/70 backdrop-blur-sm">
        <div className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              value={globalInput}
              onChange={(e) => setGlobalInput(e.target.value)}
              onKeyDown={handleGlobalInputKeyDown}
              placeholder={activeAgentNode ? '💬 Message your active agent…' : '🤖 No active agent selected. Choose one from the Agent Dashboard.'}
              disabled={!activeAgentNode || sending}
              className="w-full resize-none p-3 pr-16 rounded-lg bg-gray-800 border border-gray-700 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent min-h-[48px] max-h-[128px] transition-all"
            />
            {globalInput && (
              <div className="absolute bottom-2 right-2 text-xs text-gray-500">
                {globalInput.length}/1000
              </div>
            )}
          </div>
          
          <div className="flex flex-col gap-2">
            <button
              onClick={sendGlobalMessage}
              disabled={!activeAgentNode || !globalInput.trim() || sending}
              className="px-4 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium transition-all flex items-center gap-2"
              title="Send message (Ctrl+Enter)"
            >
              {sending ? (
                <>
                  <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin"></div>
                  Sending
                </>
              ) : (
                <>
                  Send
                  <span className="text-xs opacity-70">⏎</span>
                </>
              )}
            </button>
            
            {!activeAgentNode && (
              <button
                onClick={() => setShowAgentDashboard(true)}
                className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm font-medium transition-all"
              >
                🤖 Select Agent
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
