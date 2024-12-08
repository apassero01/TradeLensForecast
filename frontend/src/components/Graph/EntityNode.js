import React from 'react';
import { Handle, Position } from 'reactflow';

const EntityNode = ({ data }) => {
  return (
    <div className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 cursor-grab active:cursor-grabbing">
      <Handle 
        type="target" 
        position={Position.Top} 
        id="top"
        style={{ background: '#4b5563' }}
      />
      <div className="text-white font-medium pointer-events-none">{data.label}</div>
      {Object.entries(data.metaData).map(([key, value]) => (
        <div key={key} className="text-sm text-gray-400 pointer-events-none">
          {key}: {value}
        </div>
      ))}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="bottom"
        style={{ background: '#4b5563' }}
      />
    </div>
  );
};

export default React.memo(EntityNode); 