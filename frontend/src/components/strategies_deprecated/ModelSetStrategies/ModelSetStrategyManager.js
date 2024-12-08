// ModelSetStrategyManager.js
import React from 'react';

import ModelSetStrategyContainer from './ModelSetStrategyContainer';
import useStrategyData from '../StrategyManager/useStrategyData';

function ModelSetStrategyManager({ sessionState, updateSessionState }) {
  const {
    availableStrategies,
    tempStrategies,
    existingStrategies,
    error,
    handleAddStrategy,
    handleRemoveTempStrategy,
    handleSubmit,
  } = useStrategyData({
    fetchAvailableEndpoint: 'http://localhost:8000/training_session/get_model_set_strategies',
    submitEndpoint: 'http://localhost:8000/training_session/post_strategy',
    sessionState,
    updateSessionState,
    strategyKey: 'ordered_model_set_strategies',
    options: {
      manageTempStrategies: true,
      manageExistingStrategies: true,
    },
  });

  return (
    <ModelSetStrategyContainer
      availableStrategies={availableStrategies}
      tempStrategies={tempStrategies}
      existingStrategies={existingStrategies}
      error={error}
      handleAddStrategy={handleAddStrategy}
      handleRemoveTempStrategy={handleRemoveTempStrategy}
      handleSubmit={handleSubmit}
    />
  );
}

export default ModelSetStrategyManager;