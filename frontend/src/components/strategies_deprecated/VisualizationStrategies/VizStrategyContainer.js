// VizStrategyContainer.js
import React from 'react';
import VizStrategyCard from './VizStrategyCard';

function VizStrategyContainer({ strategies, onSubmit }) {
  return (
    <div className="flex flex-wrap gap-2 bg-[#1e1e1e] rounded-md border border-gray-800">
      {strategies.map((strategy) => (
        <VizStrategyCard
          key={strategy.id}
          strategy={strategy}
          onSubmit={onSubmit}
        />
      ))}
    </div>
  );
}

export default VizStrategyContainer;