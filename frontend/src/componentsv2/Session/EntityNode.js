import React from 'react';

const EntityNode = ({ data }) => {
  return (
    <div className={`
      relative 
      bg-gray-800 
      rounded-lg 
      border-2 
      ${data.type === 'TRAINING_SESSION' ? 'border-blue-500' : 
        data.type === 'FEATURE_SET' ? 'border-green-500' : 
        data.type === 'MODEL_STAGE' ? 'border-purple-500' : 
        'border-gray-500'}
      p-4 
      min-w-[250px] 
      shadow-lg
    `}>
      <div className="flex items-center gap-2 mb-3">
        <div className="text-xl font-bold text-white">{data.label}</div>
        <span className={`
          text-xs px-2 py-1 rounded-full
          ${data.type === 'TRAINING_SESSION' ? 'bg-blue-500' : 
            data.type === 'FEATURE_SET' ? 'bg-green-500' : 
            data.type === 'MODEL_STAGE' ? 'bg-purple-500' : 
            'bg-gray-500'}
        `}>
          {data.type}
        </span>
      </div>

      <div className="flex gap-2 flex-wrap mt-2">
        {data.strategies.map((strategy) => (
          <button
            key={strategy}
            className="bg-gray-700 hover:bg-gray-600 text-white text-xs px-3 py-1 rounded-full"
            onClick={(e) => {
              e.stopPropagation();
              console.log(`Execute ${strategy}`);
            }}
          >
            {strategy}
          </button>
        ))}
      </div>
    </div>
  );
};

export default EntityNode; 