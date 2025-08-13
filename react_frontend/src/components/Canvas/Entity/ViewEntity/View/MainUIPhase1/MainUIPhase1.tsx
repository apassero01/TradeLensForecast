import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { nodeSelectorFamily, childrenByTypeSelector, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import EntityViewRenderer from '../ChatInterface/EntityViewRenderer';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { EntityTypes } from '../../../../Entity/EntityEnum';
type MainUIPhase1Props = {
  data?: any;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
};

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

// Simple color hash for entity_type to HSL
function colorForType(type: string): string {
  let hash = 0;
  for (let i = 0; i < type.length; i++) {
    hash = type.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 60%, 45%)`;
}

// Left: Unified Entity Explorer (recursive tree)
function EntityTreeNode({ entityId, onSelect, onReparent, visited }: {
  entityId: string;
  onSelect: (id: string) => void;
  onReparent: (childId: string, newParentId: string) => void;
  visited: Set<string>;
}) {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));
  const [expanded, setExpanded] = useState(false);
  const draggableRef = useRef<HTMLDivElement>(null);

  if (!node || !node.data) {
    return (
      <div className="px-2 py-1 text-xs text-gray-600">(missing entity)</div>
    );
  }

  // Prevent cycles in the tree rendering
  if (visited.has(entityId)) {
    return (
      <div className="px-2 py-1 text-xs text-yellow-600">(cycle detected: {entityId})</div>
    );
  }

  const type = node?.data?.entity_type || 'entity';
  let displayName = node?.data?.name || node?.data?.entity_name || type;
  // Use view component display name for views
  if (type === EntityTypes.VIEW || type === 'view') {
    const vType = node?.data?.view_component_type;
    displayName = VIEW_NAMES[vType] || vType || displayName;
  }
  const childIds: string[] = node?.data?.child_ids || [];
  const nextVisited = (() => {
    const s = new Set(visited);
    s.add(entityId);
    return s;
  })();

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

  return (
    <div className="select-none">
      <div
        ref={draggableRef}
        draggable
        onDragStart={handleDragStart}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className="flex items-center gap-2 px-2 py-1 rounded hover:bg-gray-800 cursor-pointer"
        onClick={() => onSelect(node.data.entity_id)}
        title={`${type} • ${node.data.entity_id}`}
      >
        <button
          className="w-5 text-xs text-gray-400 hover:text-gray-200"
          onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
        >
          {childIds.length > 0 ? (expanded ? '▾' : '▸') : '•'}
        </button>
        <div
          style={{ backgroundColor: colorForType(type) }}
          className="w-4 h-4 rounded-sm grid place-items-center text-[9px] font-bold text-white"
        >
          {String(type).slice(0, 1).toUpperCase()}
        </div>
        <div className="truncate text-sm text-gray-200" style={{ maxWidth: 180 }}>{displayName}</div>
      </div>
      {expanded && childIds.length > 0 && (
        <div className="ml-5 border-l border-gray-800">
          {childIds.map((cid) => (
            <EntityTreeNode key={cid} entityId={cid} onSelect={onSelect} onReparent={onReparent} visited={nextVisited} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function MainUIPhase1({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: MainUIPhase1Props) {
  // Treat the parent as the active API model when applicable
  const parentNodeState: any = useRecoilValue(nodeSelectorFamily(parentEntityId || ''));
  const parentNode: any = parentEntityId ? parentNodeState : null;
  const isApiModel = parentNode?.entity_name === 'api_model';
  // Read active agent from THIS VIEW entity (set via Agent Dashboard)
  const viewEntityState: any = useRecoilValue(nodeSelectorFamily(viewEntityId || ''));
  const activeAgentId: string | null = viewEntityState?.data?.active_agent_id || null;
  const activeAgentNode: any = useRecoilValue(nodeSelectorFamily(activeAgentId || ''));
  // Chat/Context target preference: active agent (if valid api_model), else parent api_model, else null
  const chatApiModelNode: any = activeAgentNode?.entity_name === 'api_model' ? activeAgentNode : (isApiModel ? parentNode : null);
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);

  // Selection state for main content
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(parentEntityId || null);
  const [selectedInitialViewId, setSelectedInitialViewId] = useState<string | null>(null);
  const [showAgentDashboard, setShowAgentDashboard] = useState<boolean>(false);

  // Find parent view children to avoid rendering this view recursively
  const parentViewChildren = useRecoilValue(childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.VIEW })) as any[];
  const normalizedParentViewChildren = useMemo(() => parentViewChildren || [], [parentViewChildren]);
  const safeInitialViewId = useMemo(() => {
    const ids = normalizedParentViewChildren.map((v: any) => v?.data?.entity_id).filter(Boolean);
    // Prefer a child view that is not this view; fallback to null without causing re-renders
    const alt = ids.find((id: string) => id !== viewEntityId);
    return alt || null;
  }, [normalizedParentViewChildren, viewEntityId]);

  // Pinned and Working Context state read from active api_model
  const pinnedIds: string[] = useMemo(() => Array.isArray(chatApiModelNode?.data?.pinned_entity_ids) ? chatApiModelNode?.data?.pinned_entity_ids : [], [chatApiModelNode?.data?.pinned_entity_ids]);
  const workingIds: string[] = useMemo(() => {
    const visibles = Array.isArray(chatApiModelNode?.data?.visible_entities) ? chatApiModelNode?.data?.visible_entities : [];
    return visibles.filter((id: string) => !pinnedIds.includes(id));
  }, [chatApiModelNode?.data?.visible_entities, pinnedIds]);

  // Reparent handler is implemented via async version using useRecoilCallback

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

  const handleSelectEntity = useCallback(async (entityId: string) => {
    setShowAgentDashboard(false);
    setSelectedInitialViewId(null);
    const node = await getNode(entityId);
    const type = node?.data?.entity_type;
    // If clicking api_model
    if (node?.data?.entity_name === 'api_model') {
      if (entityId === parentEntityId) {
        // Avoid recursion: open Agent Dashboard instead of re-rendering self
        setShowAgentDashboard(true);
        setSelectedEntityId(entityId);
        return;
      }
      // Different agent: show its chat view
      const chatViewId = await getViewChildOfType(entityId, 'chatinterface');
      setSelectedEntityId(entityId);
      if (chatViewId) setSelectedInitialViewId(chatViewId);
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
  }, [getNode, getViewChildOfType, parentEntityId]);

  const handleReparentAsync = useCallback(async (childId: string, newParentId: string) => {
    if (!childId || !newParentId) return;
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

  // Pinned/Working actions
  const pinEntity = useCallback((entityId: string) => {
    if (!chatApiModelNode) return;
    const nextPinned = Array.from(new Set([...(pinnedIds || []), entityId]));
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(chatApiModelNode.entity_id)
        .withParams({ attribute_map: { pinned_entity_ids: nextPinned } })
        .withAddToHistory(false)
        .build()
    );
  }, [chatApiModelNode, pinnedIds, sendStrategyRequest]);

  const removeFromWorking = useCallback((entityId: string) => {
    if (!chatApiModelNode) return;
    const visibles: string[] = chatApiModelNode?.data?.visible_entities || [];
    const next = visibles.filter((id) => id !== entityId);
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(chatApiModelNode.entity_id)
        .withParams({ attribute_map: { visible_entities: next } })
        .withAddToHistory(false)
        .build()
    );
  }, [chatApiModelNode, sendStrategyRequest]);

  // Drag to pin: Working item -> Pinned area
  const handlePinnedDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  const handlePinnedDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const draggedId = e.dataTransfer.getData('application/x-entity-id');
    if (draggedId) pinEntity(draggedId);
  };

  // Agent Dashboard (Yellow Doc): Cards with Set Active
  const AgentDashboard: React.FC = () => {
    const agentEntities = allEntities.filter((e: any) => e?.data?.entity_name === 'api_model');
    return (
      <div className="p-4 grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {agentEntities.map((agent: any) => (
          <div key={agent.data.entity_id} className="bg-gray-800 border border-gray-700 rounded-lg p-4 flex flex-col gap-3">
            <div className="text-sm text-gray-400">API Model</div>
            <div className="text-lg font-semibold text-white">{agent.data.name || agent.data.entity_id.slice(0,8)}</div>
            <div className="text-xs text-gray-500 break-all">{agent.data.entity_id}</div>
            <div className="flex gap-2 mt-2">
              <button
                className="px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-sm"
                onClick={() => {
                  // Set as Active on THIS VIEW entity; store agent id for routing chat/context
                  sendStrategyRequest(
                    StrategyRequests.builder()
                      .withStrategyName('SetAttributesStrategy')
                      .withTargetEntity(viewEntityId)
                      .withParams({ attribute_map: { active_agent_id: agent.data.entity_id } })
                      .withAddToHistory(false)
                      .build()
                  );
                }}
              >
                Set as Active
              </button>
              <button
                className="px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-sm"
                onClick={() => {
                  // Preview chat view for this agent in center
                  setSelectedEntityId(agent.data.entity_id);
                  setShowAgentDashboard(false);
                }}
              >
                Open Chat
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Global Chat input state
  const [globalInput, setGlobalInput] = useState('');
  const [sending, setSending] = useState(false);
  const sendGlobalMessage = async () => {
    if (!chatApiModelNode || !globalInput.trim() || sending) return;
    setSending(true);
    try {
      sendStrategyRequest(
        StrategyRequests.builder()
          .withStrategyName('CallApiModelStrategy')
          .withTargetEntity(chatApiModelNode.entity_id)
          .withParams({ user_input: globalInput, serialize_entities_and_strategies: true })
          .withAddToHistory(false)
          .build()
      );
      setGlobalInput('');
    } finally {
      setSending(false);
    }
  };

  // Helper to render small entity pill
  const EntityPill: React.FC<{ id: string; onClose?: () => void; draggable?: boolean }>
    = ({ id, onClose, draggable }) => {
    const n: any = useRecoilValue(nodeSelectorFamily(id || ''));
    if (!id || !n || !n.data) {
      return (
        <div className="flex items-center gap-2 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm text-gray-500">
          (missing)
        </div>
      );
    }
    const t = n?.data?.entity_type || 'entity';
    const name = n?.data?.name || n?.data?.entity_name || t;
    const handleDragStart = (e: React.DragEvent) => {
      if (!draggable) return;
      e.dataTransfer.setData('application/x-entity-id', id);
    };
    return (
      <div
        className="flex items-center gap-2 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
        draggable={draggable}
        onDragStart={handleDragStart}
        title={`${t} • ${id}`}
      >
        <div style={{ background: colorForType(t) }} className="w-3 h-3 rounded-sm" />
        <span className="truncate max-w-[140px] text-gray-200">{name}</span>
        {onClose && (
          <button className="text-gray-400 hover:text-white" onClick={onClose} aria-label="remove">
            ×
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="nodrag flex flex-col w-full h-full bg-gray-900 text-white">
      {/* Main body: Left • Center • Right */}
      <div className="flex-1 min-h-0 flex">
        {/* Left Sidebar: Entity Explorer */}
        <div className="w-72 border-r border-gray-800 flex flex-col">
          <div className="p-3 text-xs text-gray-400 border-b border-gray-800 flex items-center justify-between">
            <span>Unified Entity Explorer</span>
            <button
              className="px-2 py-1 text-xs rounded bg-gray-800 hover:bg-gray-700 border border-gray-700"
              onClick={() => setShowAgentDashboard((v) => !v)}
              title="Agent Switcher"
            >
              Agents
            </button>
          </div>
          <div className="flex-1 overflow-auto nowheel p-2">
            <EntityTreeNode
              entityId={parentEntityId}
              onSelect={handleSelectEntity}
              onReparent={handleReparentAsync}
              visited={useMemo(() => new Set<string>(), [])}
            />
          </div>
        </div>

        {/* Center: Main Content Area */}
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex-shrink-0 p-2 border-b border-gray-800 text-xs text-gray-400">
            {selectedEntityId ? `Selected: ${selectedEntityId}` : 'Select an entity'}
          </div>
          <div className="flex-1 overflow-hidden">
            {showAgentDashboard ? (
              <AgentDashboard />
            ) : selectedEntityId ? (
              selectedEntityId === parentEntityId ? (
                safeInitialViewId ? (
                  <EntityViewRenderer
                    key={`entity-${parentEntityId}-view-${safeInitialViewId}`}
                    entityId={parentEntityId}
                    initialViewId={safeInitialViewId}
                    sendStrategyRequest={sendStrategyRequest}
                    updateEntity={updateEntity}
                  />
                ) : (
                  <div className="h-full grid place-items-center text-gray-500">
                    No other views available for this entity. Select a different entity in the explorer.
                  </div>
                )
              ) : (
                <EntityViewRenderer
                  key={`entity-${selectedEntityId}`}
                  entityId={selectedEntityId}
                  initialViewId={selectedInitialViewId || undefined}
                  sendStrategyRequest={sendStrategyRequest}
                  updateEntity={updateEntity}
                />
              )
            ) : (
              <div className="h-full grid place-items-center text-gray-500">No entity selected</div>
            )}
          </div>
        </div>

        {/* Right Sidebar: Smart Context */}
        <div className="w-80 border-l border-gray-800 flex flex-col">
          <div className="p-3 border-b border-gray-800 text-xs text-gray-400">Smart Context</div>
          <div className="p-3 space-y-4 overflow-auto nowheel">
            {/* Pinned Context */}
            <div>
              <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Pinned</div>
              <div
                className="min-h-16 p-2 rounded bg-gray-900 border border-gray-800 flex flex-wrap gap-2"
                onDragOver={handlePinnedDragOver}
                onDrop={handlePinnedDrop}
                title="Drag items here to pin"
              >
                {pinnedIds.length === 0 && (
                  <div className="text-xs text-gray-600">No pinned entities</div>
                )}
                {pinnedIds.map((id) => (
                  <EntityPill key={id} id={id} />
                ))}
              </div>
            </div>

            {/* Working Context */}
            <div>
              <div className="text-xs uppercase tracking-wide text-gray-400 mb-2">Working</div>
              <div className="min-h-16 p-2 rounded bg-gray-900 border border-gray-800 flex flex-wrap gap-2">
                {workingIds.length === 0 && (
                  <div className="text-xs text-gray-600">No working context</div>
                )}
                {workingIds.map((id) => (
                  <EntityPill key={id} id={id} draggable onClose={() => removeFromWorking(id)} />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: Global Chat Input */}
      <div className="flex-shrink-0 border-t border-gray-800 p-3 bg-gray-900/70">
        <div className="flex items-end gap-2">
          <textarea
            value={globalInput}
            onChange={(e) => setGlobalInput(e.target.value)}
            placeholder={chatApiModelNode ? 'Message the active agent…' : 'No active api_model detected on this view'}
            disabled={!chatApiModelNode || sending}
            className="flex-1 resize-none p-3 rounded bg-gray-800 border border-gray-700 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 min-h-[48px] max-h-[128px]"
          />
          <button
            onClick={sendGlobalMessage}
            disabled={!chatApiModelNode || !globalInput.trim() || sending}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50"
          >
            {sending ? 'Sending…' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}


