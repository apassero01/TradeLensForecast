import React from 'react';
import { useSession } from '../../hooks/useSession';
import SessionStatus from './SessionStatus';
import SessionSelector from './SessionSelector';
import DeleteSessionButton from './DeleteSessionButton';
import { useEffect } from 'react';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
const TopBar = () => {
    const {
        sessionId,
        isActive,
        isLoading, 
        savedSessions,
        startSession,
        stopSession,
        saveSession,
        loadSession,
        deleteSession,
        fetchSavedSessions,
        error,
        setSessionError,
    } = useSession(); 

    useEffect(() => {
        fetchSavedSessions();
    }, [sessionId]);

    useWebSocketConsumer();

    React.useEffect(() => {
        if (error) {
          const timer = setTimeout(() => {
            setSessionError(null); 
          }, 5000);
          // Cleanup if error changes or component unmounts
          return () => clearTimeout(timer);
        }
      }, [error, setSessionError]);
    
      // A helper to manually dismiss the error (user clicks X)
      const dismissError = () => {
        setSessionError(null);
      };

    return (
        <div className="bg-gray-800 border-b border-gray-700">
            {/* If there's an error, show it in a small banner */}
            {error && (
                <div className="bg-red-600 text-white p-2 text-sm flex justify-between items-center">
                <span>{error}</span>
                <button
                    onClick={dismissError}
                    className="ml-4 bg-red-700 hover:bg-red-800 px-2 py-1 rounded"
                >
                    X
                </button>
                </div>
            )}
            <div className="container mx-auto px-4 h-14 flex items-center justify-between">
                {/*left section */}
                <div className="flex items-center space-x-4">
                    <h1 className="text-white font-medium"> TradeLens Forecast</h1>
                    <SessionStatus isActive={isActive} /> 
                </div>

                {/*right section */}
                <div className="flex items-center space-x-3">
                    <button
                        onClick={() => startSession()}
                        disabled = {isLoading}
                        className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Starting...' : 'Start Session'}
                    </button>
                    <button 
                        onClick={() => stopSession()}
                        disabled={isLoading}
                        className="px-3 py-1.5 text-sm bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Stop Session
                    </button>
                    <button
                        onClick={saveSession}
                        disabled={isLoading}
                        className="px-3 py-1.5 text-sm bg-green-500 text-white rounded hover:bg-green-600
                                disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Save Session
                    </button>

                    <SessionSelector
                        onSessionSelect={loadSession}
                        savedSessions={savedSessions}
                        isLoading={isLoading}
                        sessionId={sessionId}
                    />

                    <DeleteSessionButton
                        onDeleteSession={deleteSession}
                        isLoading={isLoading}
                    />
                </div>
            </div>
        </div>
    )
}

export default TopBar;