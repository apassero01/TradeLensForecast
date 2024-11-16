import React, { useState } from 'react';
import Tabs from './Tabs';
import ConfigurationInputBox from './ConfigurationInputBox';
import { saveSession, removeSession, getSessionById } from './api';
import SessionDetails from "./SessionDetails";
import PageLayout from "./containers/PageLayout";
import PreprocessingScreen from "./screens/PreprocessingScreen";

function TrainingSession() {
  const [sessionState, setSessionState] = useState({
    allModelSets: [],
    selectedModelSets: [],
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

  const updateSessionState = (key, value) => {
    setSessionState(prevState => ({
      ...prevState,
      [key]: value
    }));
  };

  const handleLoadSession = async (sessionId) => {
    setLoading(true);
    setError('');
    try {
      const sessionData = await getSessionById(sessionId);
      updateSessionState('sessionData', sessionData);
      updateSessionState('selectedModelSets', sessionData.model_set_configs);
      updateSessionState('selectedXFeatures', sessionData.X_features.map(value => ({ name: value })));
      updateSessionState('selectedYFeatures', sessionData.y_features.map(value => ({ name: value })));
      updateSessionState('start_timestamp', sessionData.start_date);
    }
    catch (error) {
      setError('Failed to load session ' + error);
    }
    finally {
      setLoading(false);
    }
  };

  const handleSaveSession = async (sessionState) => {
    setLoading(true);
    setError('');
    try {
      await saveSession(sessionState);
    }
    catch (error) {
      setError('Failed to save session ' + error);
    }
    finally {
      setLoading(false);
    }
  }

  const handleRemoveSession = async () => {
    setLoading(true);
    setError('');
    try {
      await removeSession(sessionState);
      updateSessionState('sessionData', null);
    }
    catch (error) {
      setError('Failed to remove session ' + error);
    }
    finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col min-w-screen min-h-screen overflow-hidden">
      {/* Tabs for navigation */}
      <Tabs activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Full-height container for tab content */}
      <div className="flex-grow flex w-full h-full">
        {activeTab === 'configuration' && (
          <PageLayout
            layout={[
              {
                columnWidths: [100],
                components: [
                  {
                    component: (
                      <SessionDetails
                        sessionState={sessionState}
                        updateSessionState={updateSessionState}
                        onSave={handleSaveSession}
                        onRemove={handleRemoveSession}
                        onLoad={handleLoadSession}
                        setError={setError}
                        setLoading={setLoading}
                      />
                    ),
                  },
                ],
              },
              {
                columnWidths: [100],
                components: [
                  {
                    component: (
                      <ConfigurationInputBox
                        sessionState={sessionState}
                        updateSessionState={updateSessionState}
                        setError={setError}
                        setLoading={setLoading}
                      />
                    ),
                  },
                ],
              },
            ]}
          />
        )}

        {activeTab === 'preprocessing' && (
          <PreprocessingScreen
            sessionState={sessionState}
            updateSessionState={updateSessionState}
            setError={setError}
            setLoading={setLoading}
          />
        )}
      </div>
    </div>
  );
}

export default TrainingSession;