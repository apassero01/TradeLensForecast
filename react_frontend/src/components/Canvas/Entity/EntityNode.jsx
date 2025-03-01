// src/components/Canvas/EntityNode.jsx
import React, { memo } from 'react';
import EntityNodeBase from './EntityNodeBase';

function EntityNode({ data }) {
  return (
    <EntityNodeBase 
      data={data}
      containerClassName="flex flex-col items-center justify-center p-4"
    >
      {({ entity, handleCreateChild }) => (
        <>
          <div className="font-bold">{entity.entity_name}</div>
          <div className="text-sm text-gray-300">Type: {entity.entity_type}</div>
          <div className="text-xs text-gray-400">Children: {entity.child_ids.length}</div>
          <button
            onClick={handleCreateChild}
            className="mt-2 px-2 py-1 bg-gray-700 text-gray-200 rounded hover:bg-gray-600"
          >
            Create Child
          </button>
        </>
      )}
    </EntityNodeBase>
  );
}

export default memo(EntityNode);