import { useEffect, useRef, useState, useCallback } from 'react';
import { BACKEND_WS_URL } from '../config';
import StrategyRequest from '../utils/StrategyRequest';

export default function useEntityWebSocket({ 
  sessionStarted, 
  onEntityUpdate, 
  onError,
  initialEntities = []
}) {
  const globalSocketRef = useRef(null);
  const entitySocketsRef = useRef({});
  const messageQueueRef = useRef([]);
  const reconnectTimeoutRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectAttemptsRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 5;

  // 1. Maintain a ref that always holds the latest entity list.
  // This ensures we have up-to-date values even if the parent's state changes.
  const latestEntitiesRef = useRef(initialEntities);
  useEffect(() => {
    latestEntitiesRef.current = initialEntities;
  }, [initialEntities]);

  // 2. Maintain a ref to store the previously subscribed entity IDs.
  const lastSubscribedEntityIdsRef = useRef([]);

  const processMessageQueue = useCallback(() => {
    while (messageQueueRef.current.length > 0 && globalSocketRef.current?.readyState === WebSocket.OPEN) {
      const message = messageQueueRef.current.shift();
      console.log('Processing queued message:', message);
      try {
        globalSocketRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Error sending queued message:', error);
        onError('Failed to send queued message: ' + error.message);
      }
    }
  }, [onError]);

  const sendMessageGlobal = useCallback((message) => {
    if (globalSocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('Sending global WebSocket message:', message);
      globalSocketRef.current.send(JSON.stringify(message));
    } else {
      console.log('Queueing message for when socket is ready:', message);
      messageQueueRef.current.push(message);
    }
  }, []);

  const sendMessageEntity = useCallback((entityId, message) => {
    const socket = entitySocketsRef.current[entityId];
    if (socket?.readyState === WebSocket.OPEN) {
      console.log(`Sending entity ${entityId} WebSocket message:`, message);
      socket.send(JSON.stringify(message));
    } else {
      console.warn(`Entity ${entityId} WebSocket not ready, cannot send:`, message);
    }
  }, []);

  const connectEntitySocket = useCallback((entityId) => {
    // Don't create new connection if one exists and is open/connecting
    if (entitySocketsRef.current[entityId]) {
      if (
        entitySocketsRef.current[entityId].readyState === WebSocket.OPEN ||
        entitySocketsRef.current[entityId].readyState === WebSocket.CONNECTING
      ) {
        return entitySocketsRef.current[entityId];
      }
      if (entitySocketsRef.current[entityId].readyState === WebSocket.CLOSED) {
        entitySocketsRef.current[entityId] = null;
      }
    }

    console.log(`Connecting to Entity WebSocket for entity ${entityId}`);
    const socket = new WebSocket(`ws://${BACKEND_WS_URL}/ws/entity/${entityId}/`);
    entitySocketsRef.current[entityId] = socket;

    socket.onopen = () => {
      console.log(`Entity ${entityId} WebSocket connected!`);
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        console.log(`Entity ${entityId} received WebSocket message:`, msg);

        if (msg.type === 'entity_update') {
          // For a normal update, pass along the entity data.
          onEntityUpdate({ [entityId]: msg.entity });
        } else if (msg[entityId] && msg[entityId].deleted) {
          // Handle deletion messages.
          // For example, if msg is: 
          // { "entity-id": { deleted: true, id: "entity-id" } }
          console.log(`Entity ${entityId} marked as deleted:`, msg[entityId]);
          onEntityUpdate({ [entityId]: msg[entityId] });
        }
      } catch (error) {
        console.error(`Error processing entity ${entityId} WebSocket message:`, error);
      }
    };

    socket.onerror = (err) => {
      console.error(`Entity ${entityId} WebSocket error:`, err);
    };

    socket.onclose = () => {
      console.log(`Entity ${entityId} WebSocket closed`);
      if (entitySocketsRef.current[entityId]?.readyState === WebSocket.CLOSED) {
        delete entitySocketsRef.current[entityId];
      }
    };

    return socket;
  }, [onEntityUpdate]);

  const connectGlobalSocket = useCallback(() => {
    if (!sessionStarted) {
      return null;
    }

    if (globalSocketRef.current) {
      if (globalSocketRef.current.readyState === WebSocket.OPEN || 
          globalSocketRef.current.readyState === WebSocket.CONNECTING) {
        return globalSocketRef.current;
      }
    }

    console.log('[WS] Creating new global WebSocket connection...');
    const socket = new WebSocket(`ws://${BACKEND_WS_URL}/ws/entities/`);
    globalSocketRef.current = socket;

    socket.onopen = () => {
      console.log('[WS] Global WebSocket connected!', {
        readyState: socket.readyState,
        protocol: socket.protocol,
        url: socket.url
      });
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      setTimeout(() => {
        console.log('[WS] Processing message queue...', messageQueueRef.current);
        processMessageQueue();
      }, 100);
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        console.log('[WS] Received message:', msg);

        if (msg.type === 'connected' && msg.request_subscriptions) {
          setupEntitySubscriptions(initialEntities);
          processMessageQueue();
        } else if (msg.type === 'entity_update' && msg.entities) {
          Object.keys(msg.entities).forEach(entityId => {
            if (!entitySocketsRef.current[entityId]) {
              connectEntitySocket(entityId);
            }
          });
          onEntityUpdate(msg.entities);
        } else if (msg.type === 'strategy_executed') {
          console.log('Strategy execution confirmed:', msg);
        } else if (msg.type === 'session_deleted') {
          console.log('[WS] Session deleted successfully');
          // Clear local state
          entitySocketsRef.current = {};
          messageQueueRef.current = [];
          setIsConnected(false);
          
          // Close the socket cleanly
          if (globalSocketRef.current?.readyState === WebSocket.OPEN) {
            globalSocketRef.current.close(1000, 'Session deleted');
          }
          globalSocketRef.current = null;
        } else if (msg.type === 'error') {
          console.error('[WS] WebSocket error message:', msg.message);
          onError(msg.message);
        }
      } catch (error) {
        console.error('[WS] Message processing error:', error);
      }
    };

    socket.onerror = (err) => {
      console.error('[WS] WebSocket error:', {
        error: err,
        readyState: socket.readyState,
        url: socket.url
      });
      setIsConnected(false);
    };

    socket.onclose = (event) => {
      console.log('[WS] WebSocket closed:', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
        readyState: socket.readyState
      });
      setIsConnected(false);
      
      if (sessionStarted && 
          reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS && 
          event.code !== 1000) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
        reconnectAttemptsRef.current += 1;
        
        console.log(`[WS] Scheduling reconnect attempt ${reconnectAttemptsRef.current} in ${delay}ms`);
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          if (sessionStarted) {
            console.log('[WS] Attempting reconnection...');
            globalSocketRef.current = null;
            connectGlobalSocket();
          }
        }, delay);
      }
    };

    return socket;
  }, [sessionStarted, onEntityUpdate, connectEntitySocket, initialEntities, processMessageQueue]);

  const setupEntitySubscriptions = useCallback((entityIds) => {
    // Force the conversion to an array if it isn't one already.
    const idsArray = Array.isArray(entityIds) ? entityIds : Array.from(entityIds);
    console.log('Setting up entity subscriptions:', idsArray);

    // For each entity id, connect its websocket
    idsArray.forEach(connectEntitySocket);
    
    // Send the message with an array of IDs
    sendMessageGlobal({
      command: 'subscribe_entities',
      entity_ids: idsArray
    });
  }, [connectEntitySocket, sendMessageGlobal]);

  const executeStrategy = useCallback((strategyRequest) => {
    console.log('Executing strategy:', strategyRequest);
    sendMessageGlobal({
      command: 'execute_strategy',
      strategy: strategyRequest instanceof StrategyRequest
        ? strategyRequest.toJSON()
        : strategyRequest
    });
  }, [sendMessageGlobal]);

  const deleteSession = useCallback(async () => {
    console.log('[WS] Sending delete session command');
    sendMessageGlobal({
      command: 'delete_session'
    });
  }, [sendMessageGlobal]);

  // Initialize global socket connection
  useEffect(() => {
    let socket = null;

    if (sessionStarted && !globalSocketRef.current) {
      socket = connectGlobalSocket();
    }

    return () => {
      // Only cleanup if component is actually unmounting
      if (!sessionStarted) {
        messageQueueRef.current = []; // Clear message queue on cleanup
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        
        // Clean up entity sockets
        Object.values(entitySocketsRef.current).forEach(entitySocket => {
          if (entitySocket?.readyState === WebSocket.OPEN) {
            entitySocket.close(1000, 'Component unmounting');
          }
        });
        entitySocketsRef.current = {};
        
        // Clean up global socket
        if (socket || globalSocketRef.current) {
          const currentSocket = socket || globalSocketRef.current;
          if (currentSocket.readyState === WebSocket.OPEN) {
            currentSocket.close(1000, 'Component unmounting');
          }
          globalSocketRef.current = null;
        }
      }
    };
  }, [sessionStarted, connectGlobalSocket]);

  // Add an effect to process message queue when connection status changes
  useEffect(() => {
    if (isConnected && globalSocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('Connection established, processing message queue...');
      processMessageQueue();
    }
  }, [isConnected, processMessageQueue]);

  useEffect(() => {
    const handleOnline = () => {
      console.log('[WS] Browser is back online. Attempting to reconnect global socket...');
      if (!globalSocketRef.current || globalSocketRef.current.readyState === WebSocket.CLOSED) {
        connectGlobalSocket();
      }
    };

    window.addEventListener('online', handleOnline);
    return () => {
      window.removeEventListener('online', handleOnline);
    };
  }, [connectGlobalSocket]);

  // 3. Create a function that subscribes using the latest entity list if needed.
  const subscribeWithLatestEntities = useCallback(() => {
    // Convert to an array in case a MapIterator was passed
    const currentEntityIds = Array.isArray(latestEntitiesRef.current)
      ? latestEntitiesRef.current
      : Array.from(latestEntitiesRef.current);

    if (!currentEntityIds.length) return;  // nothing to subscribe
    // Only subscribe if the current list is different from what we had before.
    if (JSON.stringify(currentEntityIds) === JSON.stringify(lastSubscribedEntityIdsRef.current)) {
      return; // No change; no need to resend the subscription.
    }
    
    console.log('[WS] Re-subscribing with the latest entity list:', currentEntityIds);
    setupEntitySubscriptions(currentEntityIds);
    lastSubscribedEntityIdsRef.current = currentEntityIds;
  }, [setupEntitySubscriptions]);

  // 4. Use an effect that triggers only when the global connection is established.
  useEffect(() => {
    if (isConnected) {
      subscribeWithLatestEntities();
    }
  }, [isConnected, subscribeWithLatestEntities]);

  useEffect(() => {
    if (!sessionStarted) {
      console.log('[WS] Session stopped. Clearing entity subscriptions and sockets.');
      // Reset subscription cache so that on restart a fresh subscription occurs.
      lastSubscribedEntityIdsRef.current = [];
      // Close any lingering entity sockets.
      Object.values(entitySocketsRef.current).forEach(socket => {
        if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
          socket.close(1000, 'Session stopped');
        }
      });
      // Clear the entitySockets map.
      entitySocketsRef.current = {};
    }
  }, [sessionStarted]);

  return { 
    sendMessageGlobal,
    sendMessageEntity,
    isConnected,
    executeStrategy,
    setupEntitySubscriptions,
    deleteSession
  };
}