// src/components/Canvas/Entity/StrategyRequestEntity.jsx
import React, { memo, useEffect, useCallback, Suspense } from 'react';
import EntityNodeBase from './EntityNodeBase';
import StrategyEditor from '../../Strategy/StrategyEditor';

function StrategyRequestEntity({ data, updateEntity }) {
  console.log('StrategyRequestEntity rendered', data);
  // Create a callback for handling the Escape key
  const handleKeyDown = useCallback((event) => {
    // Check if the key pressed is Escape
    if (event.key === 'Escape') {
      // Find the closest node element to check if we're focused within this node
      const activeElement = document.activeElement;
      const nodeElement = activeElement.closest(`[data-id="${data.entityId}"]`);

      // Only proceed if we're focused within this specific node
      if (nodeElement) {
        console.log('Escape pressed in StrategyRequestEntity:', data.entityId);

        // We'll use the updateEntity function from the render props
        // This will be called when the component is rendered
        if (updateEntity) {
          updateEntity(data.entityId, { hidden: true });
        }
      }
    }
  }, [data.entityId]);

  // Set up the event listener
  useEffect(() => {
    // Add the event listener
    document.addEventListener('keydown', handleKeyDown);

    // Clean up the event listener when the component unmounts
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  return (
    <Suspense fallback={<div>Loading registry...</div>}>
      <EntityNodeBase
        data={data}
      >
        {({ entity, updateEntity, updateLocalField, handleCreateChild, sendStrategyRequest, isLoading, setIsLoading }) => {
          return (
            <div className="flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
              <div className="h-full w-full px-6 overflow-hidden">
                <StrategyEditor
                  existingRequest={{
                    strategy_name: data.strategy_name,
                    param_config: data.param_config,
                    target_entity_id: data.target_entity_id,
                    add_to_history: data.add_to_history,
                    nested_requests: data.nested_requests,
                    entity_id: data.entity_id,
                  }}
                  entityType={data.entity_type}
                  updateEntity={updateEntity}
                  sendStrategyRequest={sendStrategyRequest}
                  isLoading={isLoading}
                  setIsLoading={setIsLoading}
                />
              </div>
            </div>
          );
        }}
      </EntityNodeBase>
    </Suspense>
  );
}

export default memo(StrategyRequestEntity);