import React from 'react';

const StrategyList = ({ strategies, entityType, onSelect }) => {
  // Get entity-specific and generic strategies
  const entityTypeKey = entityType?.toLowerCase();
  const genericStrategies = strategies['entity'] || [];
  const entityStrategies = entityTypeKey ? (strategies[entityTypeKey] || []) : [];
  const allStrategies = [...genericStrategies, ...entityStrategies];

  return (
    <div className="flex flex-col h-full">
      <div className="flex-shrink-0 px-8 py-6 border-b border-gray-700">
        <h3 className="text-xl text-white font-semibold">
          {entityType} Strategies
        </h3>
        <p className="text-sm text-gray-400 mt-1">
          Select a strategy to configure
        </p>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="space-y-3 px-8 py-6">
          {allStrategies.map((strategy) => (
            <button
              key={strategy.name}
              className="w-full text-left px-6 py-4 bg-gray-700/50 hover:bg-gray-600/50 
                       text-white rounded-lg transition-colors duration-150
                       border border-gray-600 hover:border-gray-500"
              onClick={() => onSelect(strategy)}
            >
              <div className="font-medium">{strategy.name}</div>
              <div className="text-sm text-gray-400 mt-1">
                {strategy.config.strategy_name || 'Configure and execute this strategy'}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default StrategyList; 