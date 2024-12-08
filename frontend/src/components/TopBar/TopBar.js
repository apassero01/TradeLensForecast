import React from 'react';
import SessionStatus from './SessionStatus';
import SessionSelector from './SessionSelector';

const TopBar = ({ 
  isSessionActive, 
  onStartSession,
  onStopSession,
  onSaveSession,
  onLoadSession,
  isLoading,
  savedSessions = []
}) => {
  return (
    <div className="bg-gray-800 border-b border-gray-700">  
      <div className="container mx-auto px-4 h-14 flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center space-x-4">
          <h1 className="text-white font-medium">TradeLens Forecast</h1>
          <SessionStatus isActive={isSessionActive} />
        </div>

        {/* Right section */}
        <div className="flex items-center space-x-3">
          <button
            onClick={onStartSession}
            disabled={isLoading}
            className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Starting...' : 'New Session'}
          </button>
          <button
            onClick={onStopSession}
            disabled={isLoading}
            className="px-3 py-1.5 text-sm bg-red-500 text-white rounded hover:bg-red-600 
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Stop Session
          </button>
          <button
            onClick={onSaveSession}
            disabled={isLoading}
            className="px-3 py-1.5 text-sm bg-green-500 text-white rounded hover:bg-green-600 
                     disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save Session
          </button>
          <SessionSelector 
            onSessionSelect={onLoadSession} 
            savedSessions={savedSessions}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
};

export default TopBar; 