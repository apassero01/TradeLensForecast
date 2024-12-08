// StrategyList.js
import React from 'react';

function StrategyList({ strategies, renderStrategyCard }) {
  return (
    <div className="flex overflow-x-auto space-x-2 pb-2 border-b border-gray-600">
      {strategies.map((strategy) => renderStrategyCard(strategy))}
    </div>
  );
}

export default StrategyList;