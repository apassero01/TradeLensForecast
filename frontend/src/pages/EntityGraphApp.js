// frontend/src/pages/EntityGraphApp.js
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import EntityGraph from '../components/Graph/EntityGraph';
import TopBar from '../components/TopBar/TopBar';
import { entityApi } from '../services/api';
import { strategyApi } from '../services/strategyApi';
import EntityStore from '../stores/EntityStore';
import { WebSocketProvider } from '../contexts/WebSocketContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorDisplay from '../components/ErrorDisplay';
import 'reactflow/dist/style.css';
import '../index.css';

// Create EntityStore instance outside component
const entityStore = new EntityStore();

function EntityGraphContent({ 
  sessionStarted, 
  setSessionStarted,
  error,
  setError,
  graphData,
  setGraphData,
  onEntityUpdate
}) {
  const [isLoading, setIsLoading] = useState(false);
  const [savedSessions, setSavedSessions] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [availableStrategies, setAvailableStrategies] = useState({});

  const { executeStrategy, deleteSession } = useWebSocket();

  const startNewSession = async () => {
    setIsLoading(true);
    try {
      const response = await entityApi.startSession();
      console.log('Session start response:', response);
      
      if (response.status === 'success' && response.sessionData) {
        entityStore.updateEntities(response.sessionData);
        const newGraphData = entityStore.toGraphData();
        console.log('Processed graph data:', newGraphData);
        
        setSessionStarted(true);
        setGraphData(newGraphData || { nodes: [], edges: [] });
        setError(null);
      } else {
        throw new Error('Invalid session data received');
      }
    } catch (err) {
      console.error('Session start error:', err);
      setError('Failed to start session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const stopSession = async () => {
    setIsLoading(true);
    try {
      await entityApi.stopSession();
      entityStore.clear();
      
      setSessionStarted(false);
      setGraphData({ nodes: [], edges: [] });
      setSelectedEntity(null);
      // setEntityIds(new Set());
      
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
      console.log('Save session response:', response);
      
      if (response.status === 'success') {
        // Refresh the list of saved sessions
        const sessionsResponse = await entityApi.getSavedSessions();
        setSavedSessions(sessionsResponse.sessions || []);
        console.log('Sessions updated:', sessionsResponse.sessions);
        setError(null);
      } else {
        throw new Error('Failed to save session');
      }
    } catch (err) {
      console.error('Save session error:', err);
      setError('Failed to save session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSession = async (sessionId) => {
    setIsLoading(true);
    try {
      const response = await entityApi.loadSession(sessionId);
      console.log('Load session response:', response);
      
      if (response.status === 'success' && response.session_data) {
        entityStore.updateEntities(response.session_data);
        const graphData = entityStore.toGraphData();
        
        setSessionStarted(true);
        setGraphData(graphData);
        setError(null);
      } else {
        throw new Error('Invalid session data received');
      }
    } catch (err) {
      console.error('Load session error:', err);
      setError('Failed to load session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    setIsLoading(true);
    try {
      // Send delete command via WebSocket
      await deleteSession();
      
      // Clear local state
      entityStore.clear();
      setSessionStarted(false);
      setGraphData({ nodes: [], edges: [] });
      setSelectedEntity(null);
      console.log('savedSessions', savedSessions);
      setSavedSessions(prev => prev.filter(session => session.entity_id !== sessionId));
      setError(null);
      
      console.log('Session deleted successfully');
    } catch (err) {
      setError('Failed to delete session: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = useCallback((event, node) => {
    console.log('Node clicked:', node);
    setSelectedEntity(prev => prev?.id === node.id ? null : node);
  }, []);

  const handleStrategyExecute = useCallback(async (strategyRequest) => {
    try {
      setIsLoading(true);
      await executeStrategy(strategyRequest);
      setError(null);
    } catch (err) {
      setError('Failed to execute strategy: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  }, [executeStrategy, setError]);

  const handleStrategyListExecute = async (strategyRequestList) => {
    try {
      setIsLoading(true);
      const response = await strategyApi.executeStrategyList(strategyRequestList);
      
      if (response.entities) {
        console.log('Updating entities from strategy list response:', response.entities);
        Object.entries(response.entities).forEach(([entityId, entityData]) => {
          if (entityData.deleted) {
            entityStore.removeEntity(entityId);
          } else {
            entityStore.updateEntities({ [entityId]: entityData });
          }
        });
        
        const graphData = entityStore.toGraphData();
        setGraphData(graphData);
      }
      
      setError(null);
      return response;
    } catch (err) {
      setError('Failed to execute strategy list: ' + err.message);
      throw err;
    } finally {
      setIsLoading(false);
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
  // Fetch strategies once when component mounts
  useEffect(() => {
    fetchAvailableStrategies();
  }, []);

  // Fetch saved sessions when component mounts
  useEffect(() => {
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

    fetchSavedSessions();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 flex">
      <div className="flex-grow flex flex-col">
        <TopBar 
          isSessionActive={sessionStarted}
          onStartSession={startNewSession}
          onStopSession={stopSession}
          onSaveSession={saveSession}
          onLoadSession={loadSession}
          onDeleteSession={handleDeleteSession}
          isLoading={isLoading}
          savedSessions={savedSessions}
        />
        
        {error && <ErrorDisplay message={error} onClose={() => setError(null)} />}
        {isLoading && <LoadingSpinner />}

        <div className="flex-grow">
          {sessionStarted ? (
            <>
              {(!graphData?.nodes || graphData.nodes.length === 0) ? (
                <div className="flex items-center justify-center h-full text-gray-400">
                  No entities to display
                </div>
              ) : (
                <EntityGraph 
                  nodes={graphData.nodes || []}
                  edges={graphData.edges || []}
                  onNodeClick={handleNodeClick}
                  selectedEntity={selectedEntity}
                  onStrategyExecute={handleStrategyExecute}
                  onStrategyListExecute={handleStrategyListExecute}
                  availableStrategies={availableStrategies}
                  getAvailableStrategies={fetchAvailableStrategies}
                />
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              Start a session to view the entity graph
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const EntityGraphApp = () => {
  const [error, setError] = useState(null);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [entityIds, setEntityIds] = useState(new Set(entityStore.entities.keys()));

  // Single, consolidated handler for WebSocket entity updates.
  const handleWebSocketEntityUpdate = useCallback((updatedEntities) => {
    console.log('WS entity update:', updatedEntities);
    Object.entries(updatedEntities).forEach(([entityId, entityData]) => {
      if (entityData.deleted) {
        console.log(`Removing entity: ${entityId}`);
        entityStore.removeEntity(entityId);
      } else {
        console.log(`Updating entity: ${entityId}`);
        entityStore.updateEntities({ [entityId]: entityData });
      }
    });

    // Refresh graph and state.
    const newGraphData = entityStore.toGraphData();
    console.log('New graph data:', newGraphData);
    setGraphData(newGraphData || { nodes: [], edges: [] });
    setEntityIds(new Set(entityStore.entities.keys()));
  }, []);

  return (
    <WebSocketProvider
      sessionStarted={sessionStarted}
      onEntityUpdate={handleWebSocketEntityUpdate}
      onError={setError}
      currentEntities={entityIds}
    >
      <EntityGraphContent 
        sessionStarted={sessionStarted}
        setSessionStarted={setSessionStarted}
        error={error}
        setError={setError}
        graphData={graphData}
        setGraphData={setGraphData}
        onEntityUpdate={handleWebSocketEntityUpdate}
      />
      {error && (
        <ErrorDisplay 
          message={error} 
          onClose={() => setError(null)}
        />
      )}
    </WebSocketProvider>
  );
};

export default EntityGraphApp;
