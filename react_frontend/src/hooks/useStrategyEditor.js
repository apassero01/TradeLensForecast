// src/hooks/useStrategyEditor.js
import { useState, useEffect } from 'react';
import { useStrategyRegistry } from './useStrategyRegistry';
import { useWebSocket } from './useWebSocket';

export function useStrategyEditor(existingRequest) {
  const { registry, loading: registryLoading, error: registryError } = useStrategyRegistry();
  const { sendStrategyRequest } = useWebSocket();

  const [requestObj, setRequestObj] = useState(() => {
    if (existingRequest) return { ...existingRequest };
    return {
      strategy_name: '',
      param_config: {},
      // target_entity_id might already be set in existingRequest if needed
      add_to_history: false,
      nested_requests: [],
      entity_id: '',
    };
  });

  useEffect(() => {
    // If strategy_name is set, optionally merge example config if param_config is empty
    if (requestObj.strategy_name && registry[requestObj.strategy_name]) {
      const example = registry[requestObj.strategy_name].example_config || {};
      if (Object.keys(requestObj.param_config).length === 0) {
        setRequestObj((prev) => ({
          ...prev,
          param_config: { ...example },
          entity_id: existingRequest.entity_id,
        }));
      }
    }
  }, [requestObj.strategy_name, registry, requestObj.param_config]);

  function executeStrategy() {
    if (!requestObj.strategy_name) {
      console.warn('No strategy selected');
      return;
    }
    // We rely on the user or existingRequest to supply target_entity_id if needed
    sendStrategyRequest(requestObj);
  }

  return {
    requestObj,
    setRequestObj,
    registry,
    registryLoading,
    registryError,
    executeStrategy,
  };
}