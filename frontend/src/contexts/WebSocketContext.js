import React, { createContext, useContext, useMemo } from 'react';
import useEntityWebSocket from '../hooks/useEntityWebSocket';

const WebSocketContext = createContext(null);

export function WebSocketProvider({ children, sessionStarted, onEntityUpdate, currentEntities, onError }) {
  const webSocket = useEntityWebSocket({
    sessionStarted,
    onEntityUpdate,
    onError,
    initialEntities: currentEntities,
  });

  const value = useMemo(() => webSocket, [webSocket]);

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
} 