import React, { useState } from 'react';
import Tabs from './Tabs';
import TrainingConfiguration from './TrainingConfiguration';

function TrainingSession() {
  // Centralized state for the training session
  const [sessionState, setSessionState] = useState({
    allSequenceSets: [],
    selectedSets: [],
    allXFeatures: [],
    selectedXFeatures: [],
    allYFeatures: [],
    selectedYFeatures: [],
    start_timestamp: null,
    sessionData: null
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('training');

  // Generalized function to update session state
  const updateSessionState = (key, value) => {
    setSessionState(prevState => ({
      ...prevState,  // Keep the previous state
      [key]: value   // Update only the specific key
    }));
  };

  return (
    <div className="container mx-auto p-2 px-0">
      {/* Tabs at the top */}
      <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Session Data Display */}
      {sessionState.sessionData && (
        <div className="mt-6 bg-white shadow-lg rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Session Created</h2>
          <p className="text-gray-700">
            <strong>Session ID:</strong> {sessionState.sessionData.session_id}
          </p>
          <p className="text-gray-700">
            <strong>Created At:</strong> {sessionState.sessionData.created_at}
          </p>
          <p className="text-gray-700">
            <strong>Status:</strong> {sessionState.sessionData.status}
          </p>
        </div>
      )}

      {/* Tab Content */}
      <div className="bg-gray-100 p-6 rounded-lg shadow-md mt-4"> {/* Single Column for Tab Content */}
        {activeTab === 'configuration' && (
          <TrainingConfiguration
            sessionState={sessionState}
            updateSessionState={updateSessionState}
            setError={setError}
            setLoading={setLoading}
          />
        )}
        {activeTab === 'preprocessing' && (
          <div className="text-gray-600">
            {/* Preprocessing content placeholder */}
            <p>Preprocessing steps will go here.</p>
          </div>
        )}
      </div>

      {/* Error and Loading Indicators */}
      {error && <p className="text-red-500 mt-4">{error}</p>}
      {loading && <p className="mt-4">Loading...</p>}
    </div>
  );
}

export default TrainingSession;