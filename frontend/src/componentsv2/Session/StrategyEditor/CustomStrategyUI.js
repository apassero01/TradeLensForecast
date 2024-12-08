import React from 'react';
import TimeSeriesConfig from './custom-uis/TimeSeriesConfig';

const CustomStrategyUI = ({ strategy, config, onChange }) => {
  // For now, only implement TimeSeriesConfig
  const strategyComponents = {
    'CreateTimeSeries': TimeSeriesConfig
  };

  const StrategyComponent = strategyComponents[strategy];

  if (!StrategyComponent) {
    return (
      <div className="text-gray-400 p-4 text-center">
        No custom UI available for this strategy
      </div>
    );
  }

  return <StrategyComponent config={config} onChange={onChange} />;
};

export default CustomStrategyUI; 