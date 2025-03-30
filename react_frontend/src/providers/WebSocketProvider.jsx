// WebSocketProvider.jsx
import React, { useEffect, useRef, useCallback } from 'react';
import { useRecoilState, useSetRecoilState } from 'recoil';
import { webSocketAtom } from '../state/webSocketAtom';
import { notificationAtom } from '../state/notificationAtom';
import { useSession } from '../hooks/useSession';
import { useEntities } from '../hooks/useEntities';
import { BACKEND_WS_URL } from '../utils/config';

export const WebSocketProvider = ({ children }) => {
  // Get the current session active status
  const { isActive } = useSession();
  // Global state for the websocket connection
  const [wsGlobal, setWsGlobal] = useRecoilState(webSocketAtom);
  const { mergeEntities } = useEntities();
  const setNotification = useSetRecoilState(notificationAtom);

  // Use refs to store the websocket instance and reconnect attempt count.
  const wsRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  // To store a pending reconnect timer so we can cancel it if needed.
  const reconnectTimeoutRef = useRef(null);

  // Function to establish a new connection.
  const connect = useCallback(() => {
    console.log('Attempting to create new WebSocket connection');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${BACKEND_WS_URL}/ws/entities/`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connection opened');
      // Reset reconnect attempts and update global state.
      reconnectAttemptsRef.current = 0;
      setWsGlobal(ws);
    };

    ws.onmessage = (event) => {
      console.log('WebSocket message received:');
      try {
        const msg = JSON.parse(event.data);
        console.log('WebSocket message received:', msg.type);
        if (msg.type === 'entity_update') {
          mergeEntities(msg.entities);
        } else if (msg.type === 'error') {
          const errorMessage =
            typeof msg.error === 'object'
              ? msg.error.message || JSON.stringify(msg.error)
              : msg.error;
          console.error('Server error:', errorMessage);
          setNotification({ message: errorMessage, type: 'error' });
        } else if (msg.type === 'connected') {
          console.log('Connection confirmed by server');
        } else {
          console.warn('Unknown message type:', msg.type);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
        setNotification({ message: 'Error processing server message', type: 'error' });
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setNotification({ message: 'Connection error. Attempting to reconnect...', type: 'error' });
    };

    ws.onclose = (event) => {
      console.log('WebSocket connection closed', event.code, event.reason);
      // Clear the global state when the socket is closed.
      setWsGlobal(null);
      // If the session is active and the closure wasn’t a clean close (code 1000),
      // schedule a reconnect using exponential backoff.
      if (isActive && event.code !== 1000) {
        const delay = Math.min(30000, Math.pow(2, reconnectAttemptsRef.current) * 1000);
        console.log(`Reconnecting in ${delay / 1000} seconds (attempt ${reconnectAttemptsRef.current + 1})`);
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current += 1;
          connect();
        }, delay);
      } else {
        // Reset reconnect attempts for a clean close or inactive session.
        reconnectAttemptsRef.current = 0;
      }
    };
  }, [isActive, setWsGlobal, mergeEntities, setNotification]);

  useEffect(() => {
    // When the session is active, establish a connection.
    if (isActive) {
      connect();
    } else {
      // If the session is not active, close any existing connection.
      if (wsRef.current) {
        console.log('Session inactive: closing websocket connection');
        wsRef.current.close(1000, 'Session ended');
        wsRef.current = null;
        setWsGlobal(null);
      }
      // Cancel any pending reconnect attempts.
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    }

    // Cleanup on unmount: close the connection and cancel pending reconnects.
    return () => {
      if (wsRef.current) {
        console.log('Cleaning up WebSocket connection');
        wsRef.current.close(1000, 'Cleaning up connection');
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [isActive, setWsGlobal, mergeEntities, setNotification, connect]);

  // Heartbeat: keep the connection alive.
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    console.log('Setting up WebSocket heartbeat');
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ command: 'ping' }));
          console.debug('Sent ping');
        } catch (err) {
          console.error('Error sending ping:', err);
        }
      }
    }, 30000);
    return () => {
      console.log('Clearing heartbeat interval');
      clearInterval(pingInterval);
    };
  }, [wsGlobal]);

  return <>{children}</>;
};