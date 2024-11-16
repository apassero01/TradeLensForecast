import React, { useState, useCallback, useEffect } from 'react';
import VizStrategyManager from "../strategies/VisualizationStrategies/VizStrategyManager";

function VisualizationContainer({
  strategyManager: StrategyManager,
  visualizationScreen: VisualizationScreen,
  sessionState,
  updateSessionState,
}) {
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [visualizationData, setVisualizationData] = useState(null);

  // Memoized handleStrategySubmit to avoid re-renders
  const handleStrategySubmit = useCallback((strategy, data) => {
    // Check for differences before updating state
    if (strategy !== selectedStrategy || data.ret_val !== visualizationData) {
      setSelectedStrategy(strategy);
      setVisualizationData(data.ret_val);
      console.log("Updating visualization with new data:", { strategy, data: data.ret_val });
    } else {
      console.log("No update needed, data is identical.");
    }
  }, [selectedStrategy, visualizationData]);

  useEffect(() => {
    console.log("Rendering VisualizationContainer with selectedStrategy and visualizationData", selectedStrategy, visualizationData);
  });

  return (
    <div className="visualization-container flex flex-col space-y-4">
      {/* Strategy Manager */}
      <VizStrategyManager
        sessionState={sessionState}
        updateSessionState={updateSessionState}
        onStrategySubmit={handleStrategySubmit}
      />

      {/* Visualization Screen */}
      {selectedStrategy && visualizationData && (
        <VisualizationScreen
          strategy={selectedStrategy}
          data={visualizationData}
        />
      )}
    </div>
  );
}

export default VisualizationContainer;