// PreprocessingScreen.js
import React from 'react';
import ModelSetStrategyManager from '../strategies/ModelSetStrategies/ModelSetStrategyManager';
import TrainingSessionCard from '../session/TrainingSessionCard';
import VizStrategyManager from "../strategies/VisualizationStrategies/VizStrategyManager";
import VisualizationContainer from "../Visualization/VisualizationContainer";
import VisualizationScreen from "../Visualization/VisualizationScreen";
import ModelStageStrategyManager from "../strategies/ModelStageStrategies/ModelStageStrategyManager";
import SessionEntityMap from "../session/SessionEntityMap";
import EntityMapCard from "../session/SessionEntityMap";
import EntityMap from "../session/EntityMap";

function ModelStageScreen({ sessionState, updateSessionState, setError, setLoading }) {
  return (
    <div className="flex min-h-screen w-full bg-[#1a1a1a] text-gray-100">
      {/* Sidebar for Training Session Card */}
      <div className="w-1/5 min-h-screen bg-[#2b2b2b] p-4 rounded-lg shadow-lg mr-4 border border-gray-700 flex-shrink-0">
        <SessionEntityMap sessionData={sessionState.sessionData} />
      </div>

      {/* Main content area for strategy managers */}
      <div className="flex flex-col flex-grow space-y-6">
        {/* Model Set Strategy Manager */}
        <ModelStageStrategyManager
          sessionState={sessionState}
          updateSessionState={updateSessionState}
        />

          <EntityMap sessionData={sessionState.sessionData} />

      </div>
    </div>
  );
}

export default ModelStageScreen;