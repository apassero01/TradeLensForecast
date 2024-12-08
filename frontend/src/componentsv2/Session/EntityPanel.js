import React from 'react';
import RelationshipPath from './RelationshipPath';

const EntityPanel = ({ node }) => {
  return (
    <div className="text-white">
      <RelationshipPath node={node} />
      
      <div className="mb-6">
        <h3 className="text-lg mb-2">Attributes</h3>
        <div className="bg-gray-900 rounded p-3">
          {Object.entries(node.data.attributes || {}).map(([key, value]) => (
            <div key={key} className="mb-2">
              <span className="text-gray-400">{key}: </span>
              <span className="text-green-400">
                {typeof value === 'object' ? JSON.stringify(value) : value.toString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-6">
        <h3 className="text-lg mb-2">Available Strategies</h3>
        <div className="flex flex-col gap-2">
          {node.data.strategies.map((strategy) => (
            <button
              key={strategy}
              className="bg-purple-700 hover:bg-purple-600 px-4 py-2 rounded text-left"
              onClick={() => console.log(`Execute ${strategy} on ${node.data.label}`)}
            >
              {strategy}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EntityPanel; 