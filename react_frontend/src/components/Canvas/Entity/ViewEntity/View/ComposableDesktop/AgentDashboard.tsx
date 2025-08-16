import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { EntityDragDropUtil, EntityDragData, useDragDrop } from '../../../../../../utils/dragDropInterface';
import { EntityExplorer } from './EntityExplorer';

interface AgentDashboardProps {
  sendStrategyRequest: (request: any) => void;
  onEntityDoubleClick: (entityId: string) => void;
  updateEntity?: (entityId: string, data: any) => void;
  viewEntityId?: string;
}

interface Message {
  type: 'ai' | 'human' | 'system' | 'tool';
  content: string;
  timestamp?: string;
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

interface AgentItemProps {
  entityId: string;
  isSelected: boolean;
  onSelect: () => void;
  onDoubleClick: () => void;
}

const AgentItem: React.FC<AgentItemProps> = ({
  entityId,
  isSelected,
  onSelect,
  onDoubleClick,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));

  const handleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onSelect();
  }, [onSelect]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDoubleClick();
  }, [onDoubleClick]);

  if (!node || !node.data) {
    return null;
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
                     'Unnamed Agent';

  return (
    <div
      className={`flex items-center gap-3 py-3 px-4 rounded-lg transition-all cursor-pointer select-none ${
        isSelected 
          ? 'bg-blue-600 text-white' 
          : 'hover:bg-gray-700 text-gray-300'
      }`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      title={`${displayName} • Single-click to select • Double-click to open chat`}
    >
      {/* Agent Icon */}
      <div className="flex-shrink-0 text-xl">
        {getEntityIcon(type)}
      </div>
      
      {/* Agent Name */}
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate">
          {displayName}
        </div>
        <div className="text-xs opacity-75 truncate">
          {entityId.slice(0, 8)}...
        </div>
      </div>
    </div>
  );
};

// Entity Pill component for context display - draggable and clickable for navigation
const EntityPill: React.FC<{ 
  id: string; 
  onClose?: () => void; 
  draggable?: boolean; 
  showNavigation?: boolean;
  onEntityDoubleClick: (entityId: string) => void;
}> = ({ id, onClose, draggable, showNavigation = false, onEntityDoubleClick }) => {
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
  
  const handleDragStart = (e: React.DragEvent) => {
    if (!draggable) return;
    
    EntityDragDropUtil.startDrag(e, {
      entityId: id,
      entityType: t,
      sourceContext: 'agent-dashboard-context',
    }, {
      dragEffect: 'move',
      dragImageText: `${getEntityIcon(t)} ${name}`,
    });
  };

  const handleClick = (e: React.MouseEvent) => {
    // Don't navigate if clicking action buttons
    if ((e.target as HTMLElement).closest('button[aria-label="remove"]') || 
        (e.target as HTMLElement).closest('button[aria-label="navigate"]')) return;
    
    // Default behavior: open entity in window
    onEntityDoubleClick(id);
  };

  const handleNavigateClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onEntityDoubleClick(id);
  };
  
  return (
    <div
      className="flex items-center gap-2 px-2 py-1 rounded border text-sm transition-all cursor-pointer hover:bg-gray-700 bg-gray-800 border-gray-700"
      draggable={draggable}
      onDragStart={handleDragStart}
      onClick={handleClick}
      title={`${t} • ${id} • Click to open in window`}
    >
      <div className="flex-shrink-0 text-sm">
        {getEntityIcon(t)}
      </div>
      <span className="truncate max-w-[100px] text-gray-200">{name}</span>
      
      {/* Navigation Arrows */}
      {showNavigation && (
        <div className="flex items-center gap-1 ml-auto">
          <button
            className="p-1 text-gray-400 hover:text-blue-400 hover:bg-gray-700 rounded transition-all"
            onClick={handleNavigateClick}
            aria-label="navigate"
            title="Open entity in window"
          >
            ↗️
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

// Enhanced Entity Item for tree explorer
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

  const handleDragStart = useCallback((e: React.DragEvent) => {
    if (!node?.data) return;
    
    const displayName = node?.data?.name || 
                       node?.data?.entity_name || 
                       node?.data?.display_name ||
                       node?.data?.title ||
                       node?.data?.attributes?.name ||
                       node?.data?.attribute_map?.name ||
                       node?.data?.meta_data?.name ||
                       node?.data?.entity_type || 
                       'Unnamed';

    EntityDragDropUtil.startDrag(e, {
      entityId,
      entityType: node.data.entity_type,
      sourceContext: 'agent-dashboard-explorer',
      sourceParentId: node.data.parent_ids?.[0],
    }, {
      dragEffect: 'move',
      dragImageText: `${getEntityIcon(node.data.entity_type)} ${displayName}`,
    });
  }, [node, entityId]);

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
        draggable
        className="flex items-center gap-1.5 py-1 px-2 rounded text-sm transition-all cursor-pointer select-none hover:bg-gray-800 text-gray-300"
        style={indentStyle}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleRightClick}
        onDragStart={handleDragStart}
        title={`${type} • ${displayName} • Double-click to open • Right-click to pin • Drag to move • ${entityId}`}
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

// Chat Messages Panel Component
const ChatMessagesPanel: React.FC<{
  messages: Message[];
  agentName?: string;
  isLoading?: boolean;
}> = ({ messages, agentName, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback((smooth = false) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: smooth ? 'smooth' : 'auto' 
      });
    }
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      scrollToBottom(true);
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [messages, scrollToBottom]);

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getMessageIcon = (type: string) => {
    switch (type) {
      case 'human': return '👤';
      case 'ai': return '🤖';
      case 'system': return '⚙️';
      case 'tool': return '🔧';
      default: return '💬';
    }
  };

  const getMessageBgColor = (type: string) => {
    switch (type) {
      case 'human': return 'bg-blue-900/50 border-blue-700/50';
      case 'ai': return 'bg-gray-800/50 border-gray-700/50';
      case 'system': return 'bg-yellow-900/50 border-yellow-700/50';
      case 'tool': return 'bg-purple-900/50 border-purple-700/50';
      default: return 'bg-gray-800/50 border-gray-700/50';
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex-shrink-0 p-3 border-b border-gray-800 flex items-center gap-2">
        <span>💬</span>
        <span className="text-sm font-medium">Chat History</span>
        {agentName && (
          <span className="text-xs text-gray-500">• {agentName}</span>
        )}
      </div>
      
      <div className="flex-1 overflow-auto p-3 space-y-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <div className="text-2xl animate-spin">⟳</div>
          </div>
        ) : messages.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-lg mb-2">💬</div>
            <div className="text-sm">No messages yet</div>
            <div className="text-xs text-gray-600 mt-1">
              Start a conversation using the chat input below
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {messages.map((message, index) => (
              <div
                key={`${index}-${message.timestamp || index}`}
                className={`p-3 rounded-lg border ${getMessageBgColor(message.type)}`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm">{getMessageIcon(message.type)}</span>
                  <span className="text-xs font-medium capitalize text-gray-300">
                    {message.type === 'ai' ? 'Assistant' : message.type}
                  </span>
                  {message.timestamp && (
                    <span className="text-xs text-gray-500 ml-auto">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  )}
                </div>
                <div className="text-sm text-gray-200 whitespace-pre-wrap">
                  {message.content}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
    </div>
  );
};

export const AgentDashboard: React.FC<AgentDashboardProps> = ({
  sendStrategyRequest,
  onEntityDoubleClick,
  updateEntity,
  viewEntityId,
}) => {
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [expandedEntities, setExpandedEntities] = useState<Set<string>>(new Set());
  const [globalInput, setGlobalInput] = useState('');
  const [sending, setSending] = useState(false);
  
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  
  // Get selected agent's data - always call the hook but handle null case
  const selectedAgent: any = useRecoilValue(nodeSelectorFamily(selectedAgentId || ''));

  // Get message history from selected agent
  const messages: Message[] = useMemo(() => {
    return selectedAgent?.data?.message_history || [];
  }, [selectedAgent?.data?.message_history]);

  const fetchAgents = useCallback(() => {
    setIsLoading(true);
    
    const request = StrategyRequests.builder()
      .withStrategyName('QueryEntitiesStrategy')
      .withParams({
        entity_type: 'api_model'
      })
      .withAddToHistory(false)
      .build();
    
    sendStrategyRequest(request);
    
    // We'll update agents when entities change in the global state
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  }, [sendStrategyRequest]);

  // Fetch all agents when component mounts
  useEffect(() => {
    if (agents.length === 0) {
      fetchAgents();
    }
  }, [agents.length, fetchAgents]);

  // Update agents list when allEntities changes
  useEffect(() => {
    const apiModelEntities = allEntities.filter(entity => 
      entity.data?.entity_type === 'api_model'
    );
    
    if (apiModelEntities.length > 0) {
      setAgents(apiModelEntities);
      setIsLoading(false);
      
      // Auto-select first agent if none selected
      if (!selectedAgentId && apiModelEntities.length > 0) {
        setSelectedAgentId(apiModelEntities[0].data.entity_id);
      }
    }
  }, [allEntities, selectedAgentId]);

  const handleAgentSelect = useCallback((agentId: string) => {
    setSelectedAgentId(agentId);
  }, []);

  const handleAgentDoubleClick = useCallback((agentId: string) => {
    // Open the agent entity itself in a new window
    onEntityDoubleClick(agentId);
  }, [onEntityDoubleClick]);

  // Context management for selected agent
  const pinnedIds: string[] = useMemo(() => 
    Array.isArray(selectedAgent?.data?.pinned_entity_ids) ? selectedAgent?.data?.pinned_entity_ids : [], 
    [selectedAgent?.data?.pinned_entity_ids]
  );
  
  const workingIds: string[] = useMemo(() => {
    const visibles = Array.isArray(selectedAgent?.data?.visible_entities) ? selectedAgent?.data?.visible_entities : [];
    return visibles.filter((id: string) => !pinnedIds.includes(id));
  }, [selectedAgent?.data?.visible_entities, pinnedIds]);

  const pinEntity = useCallback((entityId: string) => {
    if (!selectedAgent) return;
    const nextPinned = Array.from(new Set([...(pinnedIds || []), entityId]));
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(selectedAgent.data.entity_id)
        .withParams({ attribute_map: { pinned_entity_ids: nextPinned } })
        .withAddToHistory(false)
        .build()
    );
  }, [selectedAgent, pinnedIds, sendStrategyRequest]);

  const handlePinEntityFromExplorer = useCallback((entityId: string) => {
    console.log('📌 Pinning entity from explorer:', entityId);
    pinEntity(entityId);
  }, [pinEntity]);

  const removeFromWorking = useCallback((entityId: string) => {
    if (!selectedAgent) return;
    const visibles: string[] = selectedAgent?.data?.visible_entities || [];
    const next = visibles.filter((id) => id !== entityId);
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(selectedAgent.data.entity_id)
        .withParams({ attribute_map: { visible_entities: next } })
        .withAddToHistory(false)
        .build()
    );
  }, [selectedAgent, sendStrategyRequest]);

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

  // Global chat functions
  const sendGlobalMessage = async () => {
    if (!selectedAgent || !globalInput.trim() || sending) return;
    setSending(true);
    try {
      sendStrategyRequest(
        StrategyRequests.builder()
          .withStrategyName('CallApiModelStrategy')
          .withTargetEntity(selectedAgent.data.entity_id)
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

  // Standardized drop handlers for pinned context
  const pinnedDropHandler = useDragDrop({
    dropEffect: 'move',
    onDrop: (data: EntityDragData) => {
      console.log('📌 Pinning entity:', data);
      pinEntity(data.entityId);
    },
    onDragOver: (event) => {
      // Visual feedback could be added here
    },
  });

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">🤖</span>
            <h2 className="text-sm font-semibold text-white">Agent Dashboard</h2>
            <div className="text-xs text-gray-500 bg-gray-800 px-2 py-1 rounded">
              Four-Panel Layout
            </div>
          </div>
          <div className="text-xs text-gray-500">
            {selectedAgent?.data?.name || 'Select an agent to start'}
          </div>
        </div>
      </div>

      {/* Four Panel Layout: Agents • Entity Explorer • Chat Messages • Context */}
      <div className="flex-1 min-h-0 flex">
        {/* Left Panel: Agent List - Resizable */}
        <div className="w-60 min-w-[200px] max-w-[400px] border-r border-gray-800 flex flex-col resize-x overflow-hidden" style={{ resize: 'horizontal' }}>
          <div className="p-3 text-xs text-gray-400 border-b border-gray-800 flex items-center justify-between">
            <span>🤖 Agents</span>
            <button
              onClick={fetchAgents}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
              disabled={isLoading}
            >
              {isLoading ? '⟳' : '↻'}
            </button>
          </div>
          
          <div className="flex-1 overflow-auto p-2">
            {isLoading ? (
              <div className="flex items-center justify-center py-8 text-gray-500">
                <div className="text-2xl animate-spin">⟳</div>
              </div>
            ) : agents.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-2xl mb-2">🤖</div>
                <div className="text-sm">No agents found</div>
                <button
                  onClick={fetchAgents}
                  className="text-xs text-blue-400 hover:text-blue-300 mt-2 transition-colors"
                >
                  Refresh
                </button>
              </div>
            ) : (
              <div className="space-y-1">
                {agents.map((agent) => (
                  <AgentItem
                    key={agent.data.entity_id}
                    entityId={agent.data.entity_id}
                    isSelected={selectedAgentId === agent.data.entity_id}
                    onSelect={() => handleAgentSelect(agent.data.entity_id)}
                    onDoubleClick={() => handleAgentDoubleClick(agent.data.entity_id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Left-Center Panel: Entity Explorer - Resizable */}
        <div className="w-80 min-w-[250px] max-w-[500px] border-r border-gray-800 flex flex-col resize-x overflow-hidden" style={{ resize: 'horizontal' }}>
          <div className="p-3 text-xs text-gray-400 border-b border-gray-800 flex items-center gap-2">
            <span>🌳</span>
            <span>Entity Explorer</span>
            {selectedAgent && (
              <span className="text-gray-600">• {selectedAgent?.data?.name || 'Agent'}</span>
            )}
          </div>
          
          <div className="flex-1 overflow-auto">
            {!selectedAgentId ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-lg mb-2">👈</div>
                <div className="text-sm">Select an agent to explore its entities</div>
              </div>
            ) : selectedAgent ? (
              <EntityExplorer
                data={{
                  onEntityDoubleClick: onEntityDoubleClick,
                  onEntityPin: handlePinEntityFromExplorer
                }}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity || (() => {})}
                viewEntityId={viewEntityId || selectedAgentId}
                parentEntityId={selectedAgentId}
              />
            ) : (
              <div className="text-center py-8 text-gray-500">
                <div className="text-lg mb-2">❌</div>
                <div className="text-sm">Agent not found</div>
              </div>
            )}
          </div>
        </div>

        {/* Right-Center Panel: Chat Messages - Resizable */}
        <div className="flex-1 min-w-[300px] border-r border-gray-800 flex flex-col resize-x overflow-hidden" style={{ resize: 'horizontal' }}>
          <ChatMessagesPanel
            messages={messages}
            agentName={selectedAgent?.data?.name}
            isLoading={sending}
          />
        </div>

        {/* Right Panel: Smart Context Management */}
        <div className="w-80 min-w-[250px] max-w-[400px] flex flex-col">
          <div className="p-3 border-b border-gray-800 text-xs text-gray-400">🧠 Smart Context</div>
          
          <div className="p-3 space-y-4 overflow-auto flex-1">
            {!selectedAgent ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-lg mb-2">👈</div>
                <div className="text-sm">Select an agent to manage its context</div>
              </div>
            ) : (
              <>
                {/* Pinned Context */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs uppercase tracking-wide text-gray-400">📌 Pinned Context</div>
                  </div>
                  <div
                    className="min-h-16 p-3 rounded bg-gray-900 border border-gray-800 border-dashed flex flex-wrap gap-2 transition-colors hover:border-gray-700"
                    {...pinnedDropHandler}
                    title="Drag entities from anywhere to pin them permanently to this agent's context"
                  >
                    {pinnedIds.length === 0 ? (
                      <div className="w-full text-center text-xs text-gray-600 py-2">
                        <div className="mb-1">📌 No pinned entities</div>
                        <div className="text-gray-700">Drag items here to keep them always visible to your agent</div>
                      </div>
                    ) : (
                      pinnedIds.map((id) => (
                        <EntityPill 
                          key={id} 
                          id={id} 
                          showNavigation 
                          onEntityDoubleClick={onEntityDoubleClick}
                        />
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
                        <div className="text-gray-700">Your agent hasn't added any entities to its working context yet</div>
                      </div>
                    ) : (
                      workingIds.map((id) => (
                        <EntityPill 
                          key={id} 
                          id={id} 
                          draggable 
                          showNavigation 
                          onClose={() => removeFromWorking(id)}
                          onEntityDoubleClick={onEntityDoubleClick}
                        />
                      ))
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Bottom: Global Chat Input */}
      <div className="flex-shrink-0 border-t border-gray-800 p-3 bg-gray-900/70 backdrop-blur-sm">
        <div className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              value={globalInput}
              onChange={(e) => setGlobalInput(e.target.value)}
              onKeyDown={handleGlobalInputKeyDown}
              placeholder={selectedAgent ? `💬 Message ${selectedAgent?.data?.name || 'agent'}…` : '🤖 Select an agent to start chatting'}
              disabled={!selectedAgent || sending}
              className="w-full resize-none p-3 pr-16 rounded-lg bg-gray-800 border border-gray-700 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent min-h-[48px] max-h-[128px] transition-all"
            />
            {globalInput && (
              <div className="absolute bottom-2 right-2 text-xs text-gray-500">
                {globalInput.length}/1000
              </div>
            )}
          </div>
          
          <button
            onClick={sendGlobalMessage}
            disabled={!selectedAgent || !globalInput.trim() || sending}
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
        </div>
      </div>
    </div>
  );
};
