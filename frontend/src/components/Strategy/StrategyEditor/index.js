import React from 'react';
import ConfigEditor from './ConfigEditor';
import StrategyRequest from '../../../utils/StrategyRequest';

const StrategyEditor = ({
  strategy,
  historyItem,
  selectedEntity,
  onExecute,
  onBack
}) => {
  const strategyRequest = React.useMemo(() => {
    if (historyItem) {
      return new StrategyRequest({
        name: historyItem.strategy_name,
        path: historyItem.strategy_path,
        config: historyItem.param_config
      });
    }
    if (strategy && selectedEntity) {
      return new StrategyRequest({
        name: strategy.name,
        path: selectedEntity.data.path,
        config: strategy.config
      });
    }
    return null;
  }, [historyItem, strategy, selectedEntity]);

  return (
    <div className="flex flex-col h-full">
      <div className="px-8 py-6 border-b border-gray-700">
        <button
          onClick={onBack}
          className="text-gray-400 hover:text-white flex items-center gap-2
                   transition-colors duration-150"
        >
          <span>←</span>
          <span>Back to strategies</span>
        </button>
      </div>
      
      <div className="flex-grow overflow-y-auto">
        <div className="px-8 py-6">
          {strategyRequest && (
            <ConfigEditor
              key={JSON.stringify(strategyRequest)}
              strategyRequest={strategyRequest}
              onExecute={onExecute}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default StrategyEditor; 