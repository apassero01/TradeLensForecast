import React from 'react';
import { Handle, Position } from 'reactflow';
import MetadataList from './MetadataList';
import MetadataValue from './MetadataValue';

const EntityNode = ({ data }) => {
  const renderMetadataValue = (value) => {
    if (Array.isArray(value)) {
      return <MetadataList items={value} />;
    }
    return <MetadataValue value={value} />;
  };

  return (
    <div className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 cursor-grab active:cursor-grabbing min-w-[250px]">
      <Handle 
        type="target" 
        position={Position.Top} 
        id="top"
        style={{ background: '#4b5563' }}
      />
      <div className="text-white font-medium mb-2">{data.label}</div>
      <div className="space-y-1.5">
        {Object.entries(data.metaData).map(([key, value]) => (
          <div key={key} className="text-sm flex items-start gap-2">
            <span className="text-gray-400">{key}:</span>
            {renderMetadataValue(value)}
          </div>
        ))}
      </div>
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