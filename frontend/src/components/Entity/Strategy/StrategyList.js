import React, { useState } from 'react';

const StrategyList = ({ strategies, entityType, onSelect, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');

  // Get entity-specific and generic strategies
  const entityTypeKey = entityType?.toLowerCase();
  const genericStrategies = strategies['entity'] || [];
  const entityStrategies = entityTypeKey ? (strategies[entityTypeKey] || []) : [];
  const allStrategies = [...genericStrategies, ...entityStrategies];

  // Filter strategies based on search term
  const filteredStrategies = allStrategies.filter(strategy => 
    strategy.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full">
      <div className="flex-none px-4 py-3 border-b border-gray-700">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-medium text-white">
            Select a Strategy
          </h3>
          <button
            onClick={onRefresh}
            className="text-gray-400 hover:text-white p-1"
            title="Refresh strategies"
          >
            ↻
          </button>
        </div>
        {/* Search input */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search strategies..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-gray-700/50 border border-gray-600 rounded px-3 py-1.5
                     text-sm text-gray-200 placeholder-gray-400
                     focus:outline-none focus:border-gray-500"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
        <div className="space-y-1 p-2">
          {filteredStrategies.map((strategy) => (
            <button
              key={strategy.name}
              className="w-full text-left px-3 py-2 bg-gray-700/30 hover:bg-gray-600/50 
                       text-gray-200 rounded transition-colors duration-150
                       border border-transparent hover:border-gray-600"
              onClick={() => onSelect(strategy)}
            >
              <div className="text-sm truncate">
                {strategy.name}
              </div>
            </button>
          ))}
          
          {filteredStrategies.length === 0 && (
            <div className="text-sm text-gray-400 text-center py-4">
              {searchTerm ? 'No matching strategies found' : 'No strategies available'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyList; 