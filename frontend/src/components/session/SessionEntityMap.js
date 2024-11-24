import React, { useState } from 'react';

function EntityMapCard({ sessionData }) {
  const [expandedEntities, setExpandedEntities] = useState({});
  console.log(sessionData);
  const toggleExpand = (entityName) => {
    setExpandedEntities((prev) => ({
      ...prev,
      [entityName]: !prev[entityName],
    }));
  };

  const renderEntity = (entity, level = 0) => {
    const { entity_name, children = [], meta_data = {} } = entity;
    const isExpanded = expandedEntities[entity_name];

    return (
      <div key={entity_name} className={`ml-${level * 4} mb-2`}>
        <button
          onClick={() => toggleExpand(entity_name)}
          className="flex items-center text-left w-full text-blue-400 hover:text-blue-500 focus:outline-none"
        >
          <span className="mr-2">
            {isExpanded ? (
              <svg
                className="w-4 h-4 transform rotate-90"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            ) : (
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            )}
          </span>
          <span className="font-semibold">{entity_name}</span>
        </button>
        {isExpanded && (
          <div className="ml-4 mt-2 border-l border-gray-700 pl-4">
            {/* Meta Data */}
            {Object.keys(meta_data).length > 0 && (
              <div className="mb-2">
                <h4 className="text-sm font-semibold text-blue-300">Meta Data:</h4>
                <div className="text-sm text-gray-300 bg-[#2b2b2b] p-2 rounded border border-gray-700">
                  {Object.entries(meta_data).map(([key, value]) => (
                    <p key={key} className="font-mono">
                      <span className="text-gray-400">{key}:</span>{' '}
                      <span className="text-gray-300">{JSON.stringify(value)}</span>
                    </p>
                  ))}
                </div>
              </div>
            )}

            {/* Children */}
            {children.length > 0 ? (
              <div>
                <h4 className="text-sm font-semibold text-blue-300">Children:</h4>
                {children.map((child) => renderEntity(child, level + 1))}
              </div>
            ) : (
              <p className="text-sm text-gray-400 italic">No children available.</p>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-[#1e1e1e] shadow-lg rounded-lg p-6 w-full max-w-md border border-gray-700 text-gray-200">
      {!sessionData ? (
        <p className="text-gray-400">Loading entity data...</p>
      ) : (
        <div>
          <h2 className="text-2xl font-bold text-blue-500 mb-4">Entity Map</h2>
          {renderEntity(sessionData)}
        </div>
      )}
    </div>
  );
}

export default EntityMapCard;