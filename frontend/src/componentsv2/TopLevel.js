import React, { useState, useEffect, useCallback } from 'react';
import EntityGraph from '../components/Graph/EntityGraph';
import TopBar from '../components/TopBar/TopBar';
import StrategyControlPanel from '../components/Strategy/StrategyControlPanel';
import { entityApi } from '../services/api';

const TopLevel = () => {
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [savedSessions, setSavedSessions] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [availableStrategies, setAvailableStrategies] = useState({});

  const fetchGraphData = async () => {
    try {
      const response = await entityApi.getEntityGraph();
      setGraphData(response);
      setError(null);
    } catch (err) {
      setError('Failed to fetch graph data: ' + err.message);
    }
  };

  const fetchSavedSessions = async () => {
    try {
      console.log('Fetching saved sessions...');
      const response = await entityApi.getSavedSessions();
      console.log('Received saved sessions:', response);
      setSavedSessions(response.sessions || []);
    } catch (err) {
      console.error('Failed to fetch saved sessions:', err);
      setError('Failed to fetch saved sessions: ' + err.message);
    }
  };

  const fetchAvailableStrategies = async () => {
    try {
      const response = await entityApi.getStrategyRegistry();
      console.log('Available strategies:', response);
      setAvailableStrategies(response);
    } catch (err) {
      setError('Failed to fetch strategies: ' + err.message);
    }
  };

  // Fetch saved sessions and strategies on component mount
  useEffect(() => {
    fetchSavedSessions();
    fetchAvailableStrategies();
  }, []);

  const startNewSession = async () => {
    setIsLoading(true);
    try {
      await entityApi.startSession();
      setSessionStarted(true);
      await fetchGraphData();
      console.log('Session started successfully');
    } catch (err) {
      setError('Failed to start session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const stopSession = async () => {
    setIsLoading(true);
    try {
      await entityApi.stopSession();
      setSessionStarted(false);
      setGraphData(null);
      setSelectedEntity(null);
      console.log('Session stopped successfully');
    } catch (err) {
      setError('Failed to stop session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const saveSession = async () => {
    setIsLoading(true);
    try {
      const response = await entityApi.saveSession();
      console.log('Session saved successfully:', response);
      setError(null);
      await fetchSavedSessions(); // Refresh saved sessions list
    } catch (err) {
      setError('Failed to save session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSession = async (sessionId) => {
    setIsLoading(true);
    try {
      const response = await entityApi.loadSession(sessionId);
      console.log('Session loaded successfully:', response);
      setSessionStarted(true);
      setGraphData(response.session_data);
      setError(null);
    } catch (err) {
      setError('Failed to load session: ' + err.message);
      setSessionStarted(false);
      setGraphData(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = useCallback((event, node) => {
    console.log('Node clicked in TopLevel:', node);
    setSelectedEntity(node);
  }, []);

  const handleExecuteStrategy = async (entity, strategy, strategyRequest) => {
    try {
      setIsLoading(true);
      await entityApi.executeStrategy(entity.id, strategyRequest);
      await fetchGraphData();
      setError(null);
    } catch (err) {
      setError('Failed to execute strategy: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex">
      {/* Main content area */}
      <div className="flex-grow flex flex-col">
        <TopBar 
          isSessionActive={sessionStarted}
          onStartSession={startNewSession}
          onStopSession={stopSession}
          onSaveSession={saveSession}
          onLoadSession={loadSession}
          isLoading={isLoading}
          savedSessions={savedSessions}
        />
        
        {error && (
          <div className="fixed top-16 left-1/2 transform -translate-x-1/2 
                        bg-red-500/10 border border-red-500 text-red-500 
                        px-6 py-4 rounded-lg shadow-lg z-50 
                        max-w-md w-full mx-4">
            <div className="font-medium mb-1">Error</div>
            <div className="text-sm whitespace-pre-wrap">{error}</div>
          </div>
        )}

        <div className="flex-grow">
          {graphData ? (
            <EntityGraph 
              data={graphData}
              onNodeClick={handleNodeClick}
              selectedEntity={selectedEntity}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              {sessionStarted ? 'Loading graph data...' : 'Start a session to view the entity graph'}
            </div>
          )}
        </div>
      </div>

      {/* Strategy Panel */}
      {sessionStarted && (
        <StrategyControlPanel
          selectedEntity={selectedEntity}
          availableStrategies={availableStrategies}
          onExecuteStrategy={handleExecuteStrategy}
        />
      )}
    </div>
  );
};

export default TopLevel; 