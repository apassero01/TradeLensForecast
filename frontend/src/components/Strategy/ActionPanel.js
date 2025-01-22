// src/components/ActionPanel.js
import React from 'react';

const ActionPanel = ({ strategyRequests, onSelectRequest }) => {
  if (!strategyRequests || strategyRequests.length === 0) {
    return (
      <div className="text-sm text-gray-400">
        No strategy requests available
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {strategyRequests.map((req, index) => (
        <div
          key={index}
          onClick={() => onSelectRequest(req)}
          className="p-2 border border-gray-700 rounded hover:bg-gray-600 cursor-pointer transition-colors"
        >
          <div className="text-white font-semibold">
            {req.strategy_name}
          </div>
          <div className="text-xs text-gray-400">
            {/* Example fields – you can omit or style as needed */}
            entity_id: {req.entity_id} &middot; target_entity_id: {req.target_entity_id}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            created_at: {String(req.created_at)}
          </div>
          <div className="text-xs text-gray-400">
            updated_at: {String(req.updated_at)}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ActionPanel;