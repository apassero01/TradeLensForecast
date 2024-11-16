import React, {useEffect, useState} from 'react';
import {getSessions} from "./api";

function SessionDetails({ sessionState, updateSessionState, onSave, onRemove, onLoad, setError, setLoading }) {
    const [sessions, setSessions] = useState([]);
    const [selectedSessionId, setSelectedSessionId] = useState('');

    useEffect(() => {
        const fetchSessions = async () => {
            setLoading(true);
            try {
                const fetchedSessions = await getSessions();
                setSessions(fetchedSessions);
            } catch (err) {
                console.log(err);
                setError('Failed to fetch sessions');
            } finally {
                setLoading(false);
            }
        };
        fetchSessions();
    }, [setLoading, setError]);

  const handleLoadSession = () => {
    if (selectedSessionId) {
      onLoad(selectedSessionId); // Call onLoad with the selected session ID to fetch session details
    }
  };

  return (
      <div className="h-full w-full bg-white shadow-lg rounded-lg p-6 flex flex-col space-y-4">
          {/* Dropdown for loading previous sessions */}
          <div className="flex items-center space-x-4">
              <select
                  value={selectedSessionId}
                  onChange={(e) => setSelectedSessionId(e.target.value)}  // Update local state for selected session ID
                  className="border border-gray-300 rounded p-2"
              >
                  <option value="">Select Previous Session</option>
                  {sessions.map((session) => (
                      <option key={session.session_id} value={session.id}>
                          {session.id}
                      </option>
                  ))}
              </select>
              <button
                  onClick={handleLoadSession}
                  className="bg-green-500 text-white px-4 py-2 rounded shadow hover:bg-green-600"
                  disabled={!selectedSessionId || sessionState.loading}
              >
                  Load Session
              </button>
          </div>

          {/* Error and Loading Messages */}
          {sessionState.error && <p className="text-red-500">{sessionState.error}</p>}
          {sessionState.loading && <p className="text-gray-500">Loading sessions...</p>}

          <div className="flex justify-between items-center">
              <div>
                  {sessionState.sessionData ? (
                      <>
                          <h2 className="text-xl font-semibold mb-4">Session Created</h2>
                          <p className="text-gray-700">
                              <strong>Session ID:</strong> {sessionState.sessionData.session_id}
                          </p>
                          <p className="text-gray-700">
                              <strong>Created At:</strong> {sessionState.sessionData.created_at || 'Not loaded yet'}
                          </p>
                          <p className="text-gray-700">
                              <strong>Status:</strong> {sessionState.sessionData.status || 'Not loaded yet'}
                          </p>
                      </>
                  ) : (
                      <p className="text-gray-700">No session details available.</p>
                  )}
              </div>
              <div>
                  {/* Save and Remove Session buttons */}
                  <button
                      onClick={() => onSave(sessionState)}
                      className="bg-blue-500 text-white px-4 py-2 rounded shadow hover:bg-blue-600 mr-2"
                  >
                      Save
                  </button>
                  <button
                      onClick={onRemove}
                      className="bg-red-500 text-white px-4 py-2 rounded shadow hover:bg-red-600"
                  >
                      Remove Session
                  </button>
              </div>
          </div>
      </div>
  );
}

export default SessionDetails;