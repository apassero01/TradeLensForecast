import React from 'react';
import { useActiveAgent } from '../../hooks/useActiveAgent';

const ActiveAgentIndicator = ({ className = '' }) => {
  const { activeAgentNode, hasActiveAgent } = useActiveAgent();

  if (!hasActiveAgent) {
    return (
      <div className={`flex items-center gap-2 text-xs text-gray-500 ${className}`}>
        <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
        <span>No active agent</span>
      </div>
    );
  }

  const agentName = activeAgentNode?.data?.name || `Agent ${activeAgentNode?.data?.entity_id?.slice(0, 8)}`;

  return (
    <div className={`flex items-center gap-2 text-xs text-gray-400 ${className}`}>
      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
      <span>Active: {agentName}</span>
    </div>
  );
};

export default ActiveAgentIndicator;
