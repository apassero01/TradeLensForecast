import React from 'react';

function EntityMap({ sessionData }) {
  const renderEntity = (entity, level = 0) => {
    const { entity_name, children = [], meta_data = {} } = entity;

    // Scaling factor for nested entities
    const scale = 1 - level * 0.1; // Reduce size by 10% per nesting level
    const minScale = 0.4; // Set a minimum size for deeply nested components

    return (
      <div
        key={entity_name}
        className={`relative border border-gray-600 bg-[#2a2a2a] rounded-lg mb-4`}
        style={{
          transform: `scale(${Math.max(scale, minScale)})`, // Scale down based on level
          transformOrigin: 'top left',
          padding: `${Math.max(16 * scale, 6)}px`, // Dynamic padding
          margin: 'auto', // Centering within the parent
          marginTop: level === 0 ? '20px' : '10px', // Extra spacing for the top-level entity
        }}
      >
        {/* Entity Title */}
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-blue-400 font-semibold text-sm">{entity_name}</h3>
          <span className="text-gray-400 italic text-xs">
            {children.length > 0 ? `${children.length} Child${children.length > 1 ? 'ren' : ''}` : 'No Children'}
          </span>
        </div>

        {/* Meta Data */}
        {Object.keys(meta_data).length > 0 && (
          <div className="mb-2">
            <h4 className="text-blue-300 text-xs font-semibold mb-1">Meta Data:</h4>
            <div className="text-gray-300 text-xs bg-[#333333] p-2 rounded-md border border-gray-700">
              {Object.entries(meta_data).map(([key, value]) => (
                <div key={key} className="mb-1">
                  <span className="text-gray-400 font-mono">{key}:</span>{' '}
                  <span className="text-gray-300">{JSON.stringify(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Children */}
        {children.length > 0 && (
          <div className="mt-2 flex flex-wrap justify-center items-start">
            {children.map((child) => renderEntity(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-[#1e1e1e] shadow-md rounded-lg p-6 w-full text-gray-200 overflow-auto">
      {sessionData && sessionData.entity_map ? (
        <div>
          <h2 className="text-2xl font-bold text-blue-500 mb-4">Entity Map</h2>
          {renderEntity(sessionData.entity_map)}
        </div>
      ) : (
        <p className="text-gray-400">Loading entity data...</p>
      )}
    </div>
  );
}

export default EntityMap;