import React, { useState } from 'react';
import StrategyEditor from './StrategyEditor';

const StrategyPanel = ({ selectedEntity, availableStrategies, onExecuteStrategy }) => {
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  
  if (!selectedEntity) return null;
  
  const strategies = availableStrategies[selectedEntity.data.label] || [];

  return (
    <div className="w-[520px] bg-gray-800 flex flex-col h-screen border-l border-gray-700">
      {!selectedStrategy ? (
        <div className="flex flex-col h-full">
          <div className="px-8 py-6 border-b border-gray-700">
            <h3 className="text-xl text-white font-semibold">
              {selectedEntity.data.label}
            </h3>
            <p className="text-sm text-gray-400 mt-1">
              Select a strategy to configure
            </p>
          </div>

          <div className="flex-grow overflow-y-auto">
            <div className="space-y-3 px-8 py-6">
              {strategies.map((strategy) => (
                <button
                  key={strategy}
                  className="w-full text-left px-6 py-4 bg-gray-700/50 hover:bg-gray-600/50 
                           text-white rounded-lg transition-colors duration-150
                           border border-gray-600 hover:border-gray-500"
                  onClick={() => setSelectedStrategy(strategy)}
                >
                  <div className="font-medium">{strategy}</div>
                  <div className="text-sm text-gray-400 mt-1">
                    Configure and execute this strategy
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col h-full">
          <div className="px-8 py-6 border-b border-gray-700">
            <button
              onClick={() => setSelectedStrategy(null)}
              className="text-gray-400 hover:text-white flex items-center gap-2
                       transition-colors duration-150"
            >
              <span>←</span>
              <span>Back to strategies</span>
            </button>
          </div>
          
          <div className="flex-grow overflow-y-auto">
            <div className="px-8 py-6">
              <StrategyEditor
                strategy={selectedStrategy}
                entityType={selectedEntity.data.label}
                onSubmit={(config) => {
                  onExecuteStrategy(selectedEntity, selectedStrategy, config);
                  setSelectedStrategy(null);
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyPanel; 