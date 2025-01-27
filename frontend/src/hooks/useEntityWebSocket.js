import { useEffect, useRef, useState, useCallback } from 'react';

export default function useEntityWebSocket({ 
  sessionStarted, 
  onEntityUpdate, 
  onError,
  initialEntityIds = []
}) {
  const globalSocketRef = useRef(null);
  const entitySocketsRef = useRef({});
  const messageQueueRef = useRef([]);
  const reconnectTimeoutRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectAttemptsRef = useRef(0);
  const MAX_RECONNECT_ATTEMPTS = 5;

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
      if (entitySocketsRef.current[entityId].readyState === WebSocket.OPEN) {
        return entitySocketsRef.current[entityId];
      }
      if (entitySocketsRef.current[entityId].readyState === WebSocket.CONNECTING) {
        return entitySocketsRef.current[entityId];
      }
      // If socket exists but is closing/closed, clean it up
      entitySocketsRef.current[entityId] = null;
    }

    console.log(`Creating new WebSocket for entity ${entityId}`);
    const socket = new WebSocket(`ws://${window.location.host}/ws/entity/${entityId}/`);
    entitySocketsRef.current[entityId] = socket;

    socket.onopen = () => {
      console.log(`Entity ${entityId} WebSocket connected!`);
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        console.log(`Entity ${entityId} received WebSocket message:`, msg);

        if (msg.type === 'entity_update') {
          onEntityUpdate({ [entityId]: msg.entity });
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
      delete entitySocketsRef.current[entityId];
    };

    return socket;
  }, [onEntityUpdate]);

  const connectGlobalSocket = useCallback(() => {
    if (!sessionStarted) {
      return null;
    }

    if (globalSocketRef.current) {
      if (globalSocketRef.current.readyState === WebSocket.OPEN) {
        processMessageQueue();  // Process any queued messages if socket is already open
        return globalSocketRef.current;
      }
      if (globalSocketRef.current.readyState === WebSocket.CONNECTING) {
        return globalSocketRef.current;
      }
      globalSocketRef.current = null;
    }

    console.log('Creating new global WebSocket connection...');
    const socket = new WebSocket(`ws://${window.location.host}/ws/entities/`);
    globalSocketRef.current = socket;

    socket.onopen = () => {
      console.log('Global WebSocket connected!');
      setIsConnected(true);
      reconnectAttemptsRef.current = 0;
      // Process any queued messages after a short delay to ensure connection is ready
      setTimeout(() => {
        console.log('Processing message queue after connection...', messageQueueRef.current);
        processMessageQueue();
      }, 100);
    };

    socket.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        console.log('Received global WebSocket message:', msg);

        if (msg.type === 'connected' && msg.request_subscriptions) {
          setupEntitySubscriptions(initialEntityIds);
          // Also process any queued messages after subscriptions are set up
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
        } else if (msg.type === 'error') {
          console.error('WebSocket error:', msg.message);
          onError(msg.message);
        }
      } catch (error) {
        console.error('Error processing global WebSocket message:', error);
      }
    };

    socket.onerror = (err) => {
      console.error('Global WebSocket error:', err);
      setIsConnected(false);
    };

    socket.onclose = () => {
      console.log('Global WebSocket closed');
      setIsConnected(false);
      
      if (sessionStarted && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
        reconnectAttemptsRef.current += 1;
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          if (sessionStarted) {
            connectGlobalSocket();
          }
        }, delay);
      }
    };

    return socket;
  }, [sessionStarted, onEntityUpdate, connectEntitySocket, initialEntityIds, processMessageQueue]);

  const setupEntitySubscriptions = useCallback((entityIds) => {
    console.log('Setting up entity subscriptions:', entityIds);
    entityIds.forEach(connectEntitySocket);
    sendMessageGlobal({
      command: 'subscribe_entities',
      entity_ids: entityIds
    });
  }, [connectEntitySocket, sendMessageGlobal]);

  const executeStrategy = useCallback((strategyRequest) => {
    console.log('Executing strategy:', strategyRequest);
    sendMessageGlobal({
      command: 'execute_strategy',
      strategy: {
        strategy_name: strategyRequest.strategy_name,
        param_config: strategyRequest.param_config,
        target_entity_id: strategyRequest.target_entity_id,
        add_to_history: strategyRequest.add_to_history,
        nested_requests: strategyRequest.nested_requests || []
      }
    });
  }, [sendMessageGlobal]);

  // Initialize global socket connection
  useEffect(() => {
    let socket = null;

    if (sessionStarted && !globalSocketRef.current) {
      socket = connectGlobalSocket();
    }

    return () => {
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
    };
  }, [sessionStarted, connectGlobalSocket]);

  // Add an effect to process message queue when connection status changes
  useEffect(() => {
    if (isConnected && globalSocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('Connection established, processing message queue...');
      processMessageQueue();
    }
  }, [isConnected, processMessageQueue]);

  return { 
    sendMessageGlobal,
    sendMessageEntity,
    isConnected,
    executeStrategy,
    setupEntitySubscriptions
  };
}