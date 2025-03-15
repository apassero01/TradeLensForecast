// src/components/Canvas/EntityNode.jsx
import React, { memo } from 'react';
import EntityNodeBase from './EntityNodeBase';
import MetadataList from './MetadataComponents/MetadataList';
import MetadataValue from './MetadataComponents/MetadataValue';

function EntityNode({ data }) {
  const metadata = data.meta_data || data.metaData || {};
  return (
    <EntityNodeBase 
      data={data}
      containerClassName="flex flex-col items-center justify-center p-4"
    >
      {({ data, handleCreateChild }) => (
        <div className="flex flex-col p-4">
          <div className="text-sm text-gray-300">Type: {data.entity_type}</div>
          <div className="text-xs text-gray-400">Children: {data.child_ids.length}</div>
          {Object.entries(metadata).map(([key, value]) => (
            <div 
              key={key} 
              className="text-sm flex items-start gap-2 rounded px-2 -mx-2"
            >
              <span className="text-gray-400">{key}:</span>
              {Array.isArray(value)
                ? <div className="relative z-20"><MetadataList items={value} /></div>
                : <MetadataValue value={value} />}
            </div>
          ))}
        </div>
      )}
    </EntityNodeBase>
  );
}

export default memo(EntityNode);