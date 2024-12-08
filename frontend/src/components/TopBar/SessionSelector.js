import React, { useState } from 'react';

const SessionSelector = ({ onSessionSelect, savedSessions, isLoading }) => {
  const [selectedSession, setSelectedSession] = useState('');

  console.log('Current saved sessions:', savedSessions);

  return (
    <div className="flex items-center space-x-2">
      <select
        value={selectedSession}
        onChange={(e) => setSelectedSession(e.target.value)}
        className="bg-gray-700 text-white text-sm rounded px-2 py-1.5 border border-gray-600"
        disabled={isLoading}
      >
        <option value="">Select Session ({savedSessions.length} available)</option>
        {savedSessions.map((session) => (
          <option key={session.id} value={session.id}>
            Session {session.id} ({new Date(session.created_at).toLocaleString()})
          </option>
        ))}
      </select>
      <button
        onClick={() => onSessionSelect(selectedSession)}
        disabled={!selectedSession || isLoading}
        className="px-3 py-1.5 text-sm bg-gray-700 text-white rounded hover:bg-gray-600 
                 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Load
      </button>
    </div>
  );
};

export default SessionSelector; 