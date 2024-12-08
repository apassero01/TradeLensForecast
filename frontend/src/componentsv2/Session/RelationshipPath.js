import React from 'react';

const RelationshipPath = ({ node }) => {
  return (
    <div className="bg-gray-700 px-3 py-2 rounded-lg text-sm mb-4">
      <div className="text-gray-400">Entity Path:</div>
      <div className="flex items-center gap-2 mt-1">
        <span className={`
          px-2 py-1 rounded
          ${node.data.type === 'TRAINING_SESSION' ? 'bg-blue-900' : 
            node.data.type === 'FEATURE_SET' ? 'bg-green-900' : 
            node.data.type === 'MODEL_STAGE' ? 'bg-purple-900' : 
            'bg-gray-900'}
        `}>
          {node.data.label}
        </span>
      </div>
    </div>
  );
};

export default RelationshipPath; 