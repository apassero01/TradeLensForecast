import { useRecoilValue } from 'recoil';
import { webSocketAtom } from '../state/webSocketAtom';
import { useCallback } from 'react';
import { useSetRecoilState } from 'recoil';
import { notificationAtom } from '../state/notificationAtom';

export function useWebSocketConsumer() {
  const ws = useRecoilValue(webSocketAtom);
  const setNotification = useSetRecoilState(notificationAtom);
  
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
      strategy: strategyRequest,
    };

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