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
            entityStore.updateEntities(response.sessionData);
            const graphData = entityStore.toGraphData();
            
            setSessionStarted(true);
            setGraphData(graphData);
            setError(null);
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
            console.log('Save session not implemented yet');
        } catch (err) {
            setError('Failed to save session: ' + err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const loadSession = async (sessionId) => {
        setIsLoading(true);
        try {
            console.log('Load session not implemented yet');
        } catch (err) {
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
            console.log('Strategy execution not implemented yet');
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
            console.log('Strategy list execution not implemented yet');
        } catch (err) {
            setError('Failed to execute strategy list: ' + err.message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    };

    const fetchAvailableStrategies = async () => {
        try {
            console.log('Fetch available strategies not implemented yet');
        } catch (err) {
            setError('Failed to fetch strategies: ' + err.message);
        }
    };

    useEffect(() => {
        fetchAvailableStrategies();
    }, []);

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