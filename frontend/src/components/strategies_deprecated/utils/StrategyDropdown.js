// StrategyDropdown.js
import React from 'react';

function StrategyDropdown({ availableStrategies, onAddStrategy }) {
  return (
    <div className="absolute right-0 mt-2 bg-[#2b2b2b] text-gray-200 shadow-lg rounded p-1 max-h-40 overflow-y-auto border border-gray-600 w-48">
      {availableStrategies.map((strategy) => (
        <div
          key={strategy.id}
          className="px-2 py-1 cursor-pointer hover:bg-gray-700 rounded text-xs font-mono"
          onClick={() => onAddStrategy(strategy)}
        >
          {strategy.name}
        </div>
      ))}
    </div>
  );
}

export default StrategyDropdown;