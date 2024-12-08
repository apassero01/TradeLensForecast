import React, { useState } from 'react';
import JSONEditor from './JSONEditor';
import NestedStrategyEditor from './NestedStrategyEditor';

const StrategyEditor = ({ strategy, entityType, onSubmit }) => {
  const [config, setConfig] = useState({
    strategy_name: strategy,
    param_config: {},
    nested_requests: []
  });

  const handleSubmit = () => {
    onSubmit({
      name: config.strategy_name,
      config: {
        strategy_name: config.strategy_name,
        param_config: config.param_config,
        nested_requests: config.nested_requests
      }
    });
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-grow">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg text-white font-semibold mb-2">
              {strategy} Configuration
            </h3>
            
            <div className="bg-gray-900/50 rounded-lg">
              <JSONEditor
                value={config.param_config}
                onChange={(newConfig) => setConfig({
                  ...config,
                  param_config: newConfig
                })}
              />
            </div>
          </div>

          <div className="border-t border-gray-700 pt-6">
            <NestedStrategyEditor
              parentConfig={config}
              onUpdateParentConfig={setConfig}
              availableStrategies={[
                'CreateFeatureSet',
                'TrainModel',
                'HandleMissingValues',
                // ... other available strategies
              ]}
            />
          </div>
        </div>
      </div>

      <div className="mt-6">
        <button
          onClick={handleSubmit}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white py-3 rounded-lg
                   transition-colors duration-150 font-medium shadow-lg
                   hover:shadow-blue-500/20"
        >
          Execute Strategy
        </button>
      </div>
    </div>
  );
};

export default StrategyEditor; 