// src/components/Canvas/EntityNode.jsx
import React, { memo } from 'react';
import { useRecoilState } from 'recoil';
import { entityFamily } from '../../state/entityFamily';
import { NodeResizeControl, Handle, Position } from '@xyflow/react';
import { useWebSocket } from '../../hooks/useWebSocket';

function EntityNode({ data }) {
  // Recoil state for the entity
  const [entity] = useRecoilState(entityFamily(data.entityId));

  // Access your WebSocket function if needed
  const { sendStrategyRequest } = useWebSocket();

  // Destructure your entity fields
  const {
    entity_name = 'Untitled',
    entity_type = 'Unknown',
    child_ids = [],
    width = 200,
    height = 100,
  } = entity;

  // Example: a strategy request (optional)
  function handleCreateChild() {
    const request = {
      strategy_name: 'CreateEntityStrategy',
      target_entity_id: data.entityId,
      param_config: {
        entity_class: 'shared_utils.entities.StrategyRequestEntity.StrategyRequestEntity',
      },
      add_to_history: true,
      nested_requests: [],
    };
    sendStrategyRequest(request);
  }

  return (
    <div
      style={{
        width,
        height,
        position: 'relative',
        backgroundColor: '#1f2937', // Tailwind gray-800
        border: '1px solid #374151', // Tailwind gray-700
        borderRadius: 4,
        color: 'white',
      }}
      className="flex flex-col items-center justify-center p-4"
    >
      {/* 
        NodeResizer adds the built-in corner handles for resizing. 
        We style them with `handleClassName` to match your previous look.
      */}
      <NodeResizeControl minWidth={100} minHeight={50}>
        <ResizeIcon />
      </NodeResizeControl>

      {/* Standard React Flow handles for connections */}
      <Handle type="target" position={Position.Top} />

      <div className="font-bold">{entity_name}</div>
      <div className="text-sm text-gray-300">Type: {entity_type}</div>
      <div className="text-xs text-gray-400">Children: {child_ids.length}</div>

      <button
        onClick={handleCreateChild}
        className="mt-2 px-2 py-1 bg-gray-700 text-gray-200 rounded hover:bg-gray-600"
      >
        Create Child
      </button>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ResizeIcon() {
  return (
    <div 
      style={{ 
        position: 'absolute', 
        right: 5, 
        bottom: 5,
        padding: '2px',
        backgroundColor: '#1f2937', // Dark gray background
        border: '1px solid #4b5563', // Gray border
        borderRadius: '4px',
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        strokeWidth="2"
        stroke="#9ca3af" // Changed to gray color
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path stroke="none" d="M0 0h24v24H0z" fill="none" />
        <polyline points="16 20 20 20 20 16" />
        <line x1="14" y1="14" x2="20" y2="20" />
        <polyline points="8 4 4 4 4 8" />
        <line x1="4" y1="4" x2="10" y2="10" />
      </svg>
    </div>
  );
}
 

export default memo(EntityNode);