import React from 'react';
import JSONEditor from './JSONEditor';

const NestedStrategyEditor = ({ 
  parentConfig, 
  onUpdateParentConfig, 
  availableStrategies 
}) => {
  const addNestedStrategy = () => {
    const nestedRequests = parentConfig.nested_requests || [];
    onUpdateParentConfig({
      ...parentConfig,
      nested_requests: [
        ...nestedRequests,
        {
          strategy_name: '',
          param_config: {},
          nested_requests: []
        }
      ]
    });
  };

  const updateNestedStrategy = (index, updates) => {
    const nestedRequests = [...(parentConfig.nested_requests || [])];
    nestedRequests[index] = { ...nestedRequests[index], ...updates };
    onUpdateParentConfig({
      ...parentConfig,
      nested_requests: nestedRequests
    });
  };

  const removeNestedStrategy = (index) => {
    const nestedRequests = [...(parentConfig.nested_requests || [])];
    nestedRequests.splice(index, 1);
    onUpdateParentConfig({
      ...parentConfig,
      nested_requests: nestedRequests
    });
  };

  return (
    <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2">
      <div className="flex justify-between items-center sticky top-0 bg-gray-800 py-2 z-10">
        <h4 className="text-white text-sm font-medium">Nested Strategies</h4>
        <button
          onClick={addNestedStrategy}
          className="px-3 py-1 bg-blue-500 hover:bg-blue-600 rounded text-white text-sm 
                   transition-colors duration-150 ease-in-out flex items-center gap-2"
        >
          <span>+</span>
          <span>Add Nested Strategy</span>
        </button>
      </div>

      <div className="space-y-4">
        {(parentConfig.nested_requests || []).map((nestedStrategy, index) => (
          <div 
            key={index} 
            className="pl-4 border-l-2 border-gray-700 space-y-2 relative
                     before:content-[''] before:absolute before:left-0 before:top-0 
                     before:w-4 before:h-[2px] before:bg-gray-700"
          >
            <div className="flex justify-between items-center gap-2 mb-3">
              <select
                value={nestedStrategy.strategy_name}
                onChange={(e) => updateNestedStrategy(index, { 
                  strategy_name: e.target.value,
                  param_config: {} 
                })}
                className="bg-gray-700 text-white rounded px-3 py-2 flex-grow
                         border border-gray-600 focus:border-blue-500 
                         focus:ring-1 focus:ring-blue-500 outline-none"
              >
                <option value="">Select Strategy</option>
                {availableStrategies.map(strategy => (
                  <option key={strategy} value={strategy}>
                    {strategy}
                  </option>
                ))}
              </select>
              <button
                onClick={() => removeNestedStrategy(index)}
                className="text-red-400 hover:text-red-300 px-2 py-1 rounded
                         hover:bg-red-500/10 transition-colors duration-150"
              >
                ✕
              </button>
            </div>

            {nestedStrategy.strategy_name && (
              <div className="mt-2 space-y-4">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <JSONEditor
                    value={nestedStrategy.param_config}
                    onChange={(newConfig) => updateNestedStrategy(index, { 
                      param_config: newConfig 
                    })}
                  />
                </div>
                
                <NestedStrategyEditor
                  parentConfig={nestedStrategy}
                  onUpdateParentConfig={(updates) => updateNestedStrategy(index, updates)}
                  availableStrategies={availableStrategies}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default NestedStrategyEditor; 