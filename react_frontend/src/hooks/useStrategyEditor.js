import { useRecoilValue, useRecoilState } from 'recoil';
import { registrySelector, refreshTriggerAtom } from '../state/registryState';
import { useWebSocketConsumer } from './useWebSocketConsumer';
import { useState, useEffect } from 'react';
import { startTransition } from 'react';

export function useStrategyEditor(existingRequest) {
  const { sendStrategyRequest } = useWebSocketConsumer();
  const registry = useRecoilValue(registrySelector);
  const [, setRefreshTrigger] = useRecoilState(refreshTriggerAtom);

  const [requestObj, setRequestObj] = useState(() => {
    return existingRequest
      ? { ...existingRequest }
      : {
          strategy_name: '',
          param_config: {},
          add_to_history: false,
          nested_requests: [],
          entity_id: '',
        };
  });

  const refresh = () => {
    startTransition(() => {
      setRefreshTrigger((prev) => prev + 1);
    });
  };


  function executeStrategy() {
    if (!requestObj.strategy_name) {
      console.warn('No strategy selected');
      return;
    }
    sendStrategyRequest(requestObj);
  }

  return {
    requestObj,
    setRequestObj,
    registry,
    executeStrategy,
  };
}