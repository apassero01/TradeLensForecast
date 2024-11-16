// VisualizationScreen.js

import React from 'react';
import visualizationComponents from './visualizationComponents';

function VisualizationScreen({ strategy, data }) {
  const chartType = strategy.config.graph_type.toLowerCase();
  console.log("here")
  console.log("chart data is", data)

  const VisualizationComponent = visualizationComponents[chartType];

  if (!VisualizationComponent) {
    return (
      <p className="text-red-500">
        No visualization available for chart type: {chartType}
      </p>
    );
  }

  return (
    <div className="visualization-screen p-4 bg-[#1e1e1e] rounded-lg">
      <VisualizationComponent dataset={data} />
    </div>
  );
}

export default VisualizationScreen;