// VizStrategyManager.js
import React, { useEffect } from 'react';
import useStrategyData from '../StrategyManager/useStrategyData';
import VizStrategyContainer from './VizStrategyContainer';

function VizStrategyManager({
  sessionState,
  updateSessionState,
  onStrategySubmit,
}) {
  const {
    availableStrategies,
    responseData,
    error,
    handleSubmit,
  } = useStrategyData({
    fetchAvailableEndpoint: 'http://localhost:8000/training_session/get_viz_processing_strategies',
    submitEndpoint: "http://localhost:8000/training_session/post_strategy",
    sessionState,
    updateSessionState,
    strategyKey: 'ordered_viz_strategies',
    options: {
      manageTempStrategies: false,
      manageExistingStrategies: false,
      returnResponseData: true, // Enable response data
    },
  });

  // Use useEffect to detect changes in responseData
  useEffect(() => {
    if (responseData && responseData.strategy && responseData.data) {
      // Pass the strategy and data up to VisualizationContainer
      console.log('Received dat1a:', responseData);
      onStrategySubmit(responseData.strategy, responseData.data);
    }
  }, [responseData, onStrategySubmit]);

  return (
    <div className="viz-strategy-manager">
      <VizStrategyContainer
        strategies={availableStrategies}
        onSubmit={handleSubmit}
      />
      {error && <p className="text-red-500">{error}</p>}
    </div>
  );
}

export default VizStrategyManager;