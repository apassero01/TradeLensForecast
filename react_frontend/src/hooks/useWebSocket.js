import { useEffect, useCallback } from 'react'; 
import { useRecoilState, useSetRecoilState } from 'recoil';
import { webSocketAtom } from '../state/webSocketAtom';
import { notificationAtom } from '../state/notificationAtom';
import { useEntities } from './useEntities';
import { useSession } from './useSession';
import { BACKEND_WS_URL } from '../utils/config';

export function useWebSocket() {
    const { isActive } = useSession(); 
    const [ws, setWs] = useRecoilState(webSocketAtom);
    const { mergeEntities } = useEntities();
    const setNotification = useSetRecoilState(notificationAtom);

    useEffect(() => {
        // Check if we already have a valid connection
        if (ws?.readyState === WebSocket.OPEN) {
            console.log('Using existing WebSocket connection');
            return;
        }

        if (!isActive) {
            console.log('Session not active, skipping WebSocket connection');
            return;
        }

        console.log('Creating new WebSocket connection');
        const wsUrl = `ws://${BACKEND_WS_URL}/ws/entities/`; 
        const newWs = new WebSocket(wsUrl); 

        newWs.onopen = () => {
            console.log('WebSocket connection opened');
            setWs(newWs);
        }

        newWs.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'entity_update') {
                    mergeEntities(msg.entities);
                }
                else if (msg.type === 'error') {
                    // Handle different error message formats
                    const errorMessage = typeof msg.error === 'object' ? 
                        msg.error.message || JSON.stringify(msg.error) : 
                        msg.error;
                    
                    console.error('Server error:', errorMessage);
                    setNotification({
                        message: errorMessage,
                        type: 'error'
                    });
                }
                else {
                    console.warn('Unknown message type:', msg.type);
                }
            }
            catch (error) {
                console.error('Error parsing WebSocket message:', error);
                setNotification({
                    message: 'Error processing server message',
                    type: 'error'
                });
            }
        };

        newWs.onerror = (err) => {
            console.error('WebSocket error:', err);
            setNotification({
                message: 'Connection error. Attempting to reconnect...',
                type: 'error'   
            });
        };

        newWs.onclose = (event) => {
            console.log('WebSocket connection closed', event.code, event.reason);
            
            // Only attempt reconnect if it wasn't a clean closure
            if (event.code !== 1000) {
                console.log('Attempting to reconnect...');
                setTimeout(() => {
                    setWs(null); // This will trigger a new connection attempt
                }, 3000);
            }
            setWs(null);
        };

        // Cleanup function
        return () => {
            if (newWs?.readyState === WebSocket.OPEN) {
                console.log('Cleaning up WebSocket connection');
                newWs.close(1000, 'Component unmounting');
            }
        };
    }, [isActive, mergeEntities, setNotification, ws, setWs]); // Only depend on connection state changes

    const sendStrategyRequest = useCallback((strategyRequest) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.error('No active WebSocket connection', ws?.readyState);
            setNotification({
                message: 'Connection lost. Please try again.',
                type: 'error'
            });
            return;
        }

        const payload = {
            command: 'execute_strategy',
            strategy: strategyRequest, // Fixed property name
        };
        console.log('Sending strategy request:', payload);

        try {
            console.log('Sending strategy request:', payload);
            ws.send(JSON.stringify(payload));
        } catch (error) {
            console.error('Error sending strategy request:', error);
            setNotification({
                message: 'Error sending request. Please try again.',
                type: 'error'
            });
        }
    }, [ws, setNotification]);

    return { 
        sendStrategyRequest,
        isConnected: ws?.readyState === WebSocket.OPEN 
        
    };
}
