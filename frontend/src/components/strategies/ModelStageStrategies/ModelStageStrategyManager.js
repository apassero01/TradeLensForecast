// ModelStageStrategyManager.js
import React from 'react';

import useStrategyData from '../StrategyManager/useStrategyData';
import ModelSetStrategyContainer from "../ModelSetStrategies/ModelSetStrategyContainer";

function ModelStageStrategyManager({ sessionState, updateSessionState }) {
  const {
    availableStrategies,
    tempStrategies,
    existingStrategies,
    error,
    handleAddStrategy,
    handleRemoveTempStrategy,
    handleSubmit,
  } = useStrategyData({
    fetchAvailableEndpoint: 'http://localhost:8000/training_session/get_strategy_registry',
    submitEndpoint: 'http://localhost:8000/training_session/post_strategy_request',
    sessionState,
    updateSessionState,
    strategyKey: 'none',
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

export default ModelStageStrategyManager;