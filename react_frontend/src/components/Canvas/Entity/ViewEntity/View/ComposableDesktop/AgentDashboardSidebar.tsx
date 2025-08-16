import React, { useState, useCallback, useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface AgentDashboardProps {
  sendStrategyRequest: (request: any) => void;
  onEntityDoubleClick: (entityId: string) => void;
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

interface ChildEntityItemProps {
  entityId: string;
  onDoubleClick: (entityId: string) => void;
}

const ChildEntityItem: React.FC<ChildEntityItemProps> = ({
  entityId,
  onDoubleClick,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDoubleClick(entityId);
  }, [entityId, onDoubleClick]);

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
                     type ||
                     'Unnamed';

  return (
    <div
      className="flex items-center gap-2 py-2 px-3 rounded-md transition-all cursor-pointer select-none hover:bg-gray-700 text-gray-300"
      onDoubleClick={handleDoubleClick}
      title={`${displayName} • Double-click to open in new window`}
    >
      {/* Child Entity Icon */}
      <div className="flex-shrink-0 text-sm">
        {getEntityIcon(type)}
      </div>
      
      {/* Child Entity Name */}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">
          {displayName}
        </div>
      </div>
      
      {/* Entity Type Badge */}
      <div className="text-[10px] text-gray-500 bg-gray-800 px-1.5 py-0.5 rounded">
        {type}
      </div>
    </div>
  );
};

export const AgentDashboard: React.FC<AgentDashboardProps> = ({
  sendStrategyRequest,
  onEntityDoubleClick,
}) => {
  const [agents, setAgents] = useState<any[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  const allEntities = useRecoilValue<any[]>(allEntitiesSelector);
  
  // Get selected agent's children - always call the hook but handle null case
  const selectedAgent: any = useRecoilValue(nodeSelectorFamily(selectedAgentId || ''));
  const selectedAgentChildren = selectedAgentId && selectedAgent?.data?.child_ids 
    ? allEntities.filter(entity => selectedAgent.data.child_ids.includes(entity.data.entity_id))
    : [];

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
    // Find the chat view child of this agent and open it
    const agent = allEntities.find(e => e.data.entity_id === agentId);
    if (agent?.data?.child_ids) {
      const chatView = allEntities.find(entity => 
        agent.data.child_ids.includes(entity.data.entity_id) && 
        (entity.data?.entity_type === 'view' || entity.data?.view_type === 'chatinterface')
      );
      
      if (chatView) {
        onEntityDoubleClick(chatView.data.entity_id);
      } else {
        // If no chat view found, just open the agent itself
        onEntityDoubleClick(agentId);
      }
    } else {
      onEntityDoubleClick(agentId);
    }
  }, [allEntities, onEntityDoubleClick]);

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="flex-shrink-0 p-3 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-lg">🤖</span>
          <h2 className="text-sm font-semibold text-white">Agent Dashboard</h2>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Select an agent to view its capabilities • Double-click to open
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Agents List */}
        <div className="w-1/2 flex flex-col border-r border-gray-700">
          <div className="flex-shrink-0 p-3 border-b border-gray-700">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-300">Agents</h3>
              <button
                onClick={fetchAgents}
                className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                disabled={isLoading}
              >
                {isLoading ? '⟳' : '↻'}
              </button>
            </div>
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

        {/* Selected Agent's Children */}
        <div className="w-1/2 flex flex-col">
          <div className="flex-shrink-0 p-3 border-b border-gray-700">
            <h3 className="text-sm font-medium text-gray-300">
              {selectedAgent?.data?.name || 'Agent'} Views
            </h3>
          </div>
          
          <div className="flex-1 overflow-auto p-2">
            {!selectedAgentId ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-lg mb-2">👈</div>
                <div className="text-sm">Select an agent to see its views</div>
              </div>
            ) : selectedAgentChildren.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <div className="text-lg mb-2">📭</div>
                <div className="text-sm">No views available</div>
              </div>
            ) : (
              <div className="space-y-1">
                {selectedAgentChildren.map((child) => (
                  <ChildEntityItem
                    key={child.data.entity_id}
                    entityId={child.data.entity_id}
                    onDoubleClick={onEntityDoubleClick}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer with Instructions */}
      <div className="flex-shrink-0 p-3 border-t border-gray-800 text-xs text-gray-500">
        <div className="space-y-1">
          <div>💡 <strong>Double-click agents</strong> to open their chat</div>
          <div>🪟 <strong>Double-click views</strong> to open in windows</div>
        </div>
      </div>
    </div>
  );
};
