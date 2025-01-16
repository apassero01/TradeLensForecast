import React, { useState, useEffect, useCallback, useMemo } from 'react';
import EntityGraph from '../components/Graph/EntityGraph';
import TopBar from '../components/TopBar/TopBar';
import StrategyPanel from '../components/Strategy/StrategyPanel';
import { entityApi } from '../services/api';
import { strategyApi } from '../services/strategyApi';
import EntityStore from '../stores/EntityStore';
import 'reactflow/dist/style.css';

const EntityGraphApp = () => {
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionStarted, setSessionStarted] = useState(false);
    const [graphData, setGraphData] = useState(null);
    const [savedSessions, setSavedSessions] = useState([]);
    const [selectedEntity, setSelectedEntity] = useState(null);
    const [availableStrategies, setAvailableStrategies] = useState({});
    const entityStore = useMemo(() => new EntityStore(), []);

    const startNewSession = async () => {
        setIsLoading(true);
        try {
            const response = await entityApi.startSession();
            console.log('Session start response:', response);
            
            if (response.status === 'success' && response.sessionData) {
                entityStore.updateEntities(response.sessionData);
                const graphData = entityStore.toGraphData();
                console.log('Processed graph data:', graphData);
                
                setSessionStarted(true);
                setGraphData(graphData);
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
            console.log('Loading session:', sessionId);
            const response = await entityApi.loadSession(sessionId);
            
            if (response.status === 'success') {
                console.log('Session loaded successfully:', response);
                
                // Update entity store with the loaded session data
                if (response.session_data) {
                    entityStore.clear();
                    entityStore.updateEntities(response.session_data);
                    const graphData = entityStore.toGraphData();
                    setGraphData(graphData);
                    setSessionStarted(true);
                }
                
                setError(null);
            } else {
                throw new Error('Failed to load session');
            }
        } catch (err) {
            console.error('Load session error:', err);
            setError('Failed to load session: ' + err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleNodeClick = useCallback((event, node) => {
        console.log('Node clicked:', node);
        setSelectedEntity(node);
    }, []);

    const handleStrategyExecute = async (strategyRequest) => {
        try {
            setIsLoading(true);
            const response = await strategyApi.executeStrategy(strategyRequest);
            console.log('Strategy response:', response);
            
            if (response.entities) {
                console.log('Updating entities from strategy response:', response.entities);
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
            setError('Failed to execute strategy: ' + err.message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    };

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
    }, []); // Empty dependency array means this runs once on mount

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
                    {sessionStarted ? (
                        <>
                            {graphData.nodes.length === 0 ? (
                                <div className="flex items-center justify-center h-full text-gray-400">
                                    No entities to display
                                </div>
                            ) : (
                                <EntityGraph 
                                    nodes={graphData.nodes}
                                    edges={graphData.edges}
                                    onNodeClick={handleNodeClick}
                                    selectedEntity={selectedEntity}
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

            {/* Strategy Panel */}
            {sessionStarted && (
                <StrategyPanel
                    selectedEntity={selectedEntity}
                    availableStrategies={availableStrategies}
                    onStrategyExecute={handleStrategyExecute}
                    fetchAvailableStrategies={fetchAvailableStrategies}
                    onStrategyListExecute={handleStrategyListExecute}
                />
            )}
        </div>
    );
};

export default EntityGraphApp; 