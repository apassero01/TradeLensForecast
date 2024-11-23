// ModelSetStrategyContainer.js
import React, { useState } from 'react';
import ModelStageStrategyCard from './ModelStageStrategyCard';
import StrategyDropdown from "../utils/StrategyDropdown";
import StrategyList from "../utils/StrategyList";

function ModelStageStrategyContainer({
  availableStrategies,
  tempStrategies,
  existingStrategies,
  error,
  handleAddStrategy,
  handleRemoveTempStrategy,
  handleSubmit,
}) {
  const [showDropdown, setShowDropdown] = useState(false);

  return (
    <div className="p-4 bg-[#1e1e1e] rounded-md shadow border border-gray-800 space-y-4">
      {/* Title and Add Strategy Button on the Same Line */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-200">Submitted Strategies</h3>
        <button
          onClick={() => setShowDropdown(!showDropdown)}
          className="px-3 py-1 bg-blue-600 text-white font-medium rounded shadow hover:bg-blue-700 transition-colors focus:outline-none text-xs"
        >
          Add Strategy
        </button>
      </div>

      {/* Strategy Dropdown */}
      {showDropdown && (
        <StrategyDropdown
          availableStrategies={availableStrategies}
          onAddStrategy={(strategy) => {
            handleAddStrategy(strategy);
            setShowDropdown(false);
          }}
        />
      )}

      {/* Existing Strategies */}
      {existingStrategies && existingStrategies.length > 0 && (
        <div className="space-y-2">
          <StrategyList
            strategies={existingStrategies}
            renderStrategyCard={(strategy) => (
              <ModelStageStrategyCard
                key={strategy.id}
                strategy={strategy}
                onSubmit={handleSubmit}
                isSubmitted={true}
              />
            )}
          />
        </div>
      )}

      {/* Unsubmitted Strategies */}
      {tempStrategies && tempStrategies.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-200 mb-1">Unsubmitted Strategies</h3>
          <StrategyList
            strategies={tempStrategies}
            renderStrategyCard={(strategy) => (
              <ModelStageStrategyCard
                key={strategy.id}
                strategy={strategy}
                onSubmit={handleSubmit}
                onRemove={() => handleRemoveTempStrategy(strategy.id)}
                isSubmitted={false}
              />
            )}
          />
        </div>
      )}

      {/* Error Message */}
      {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
    </div>
  );
}

export default ModelStageStrategyContainer;