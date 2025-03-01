// src/components/Canvas/EntityNodeBase.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { useRecoilState, useRecoilValue, useRecoilCallback } from 'recoil';
import { entityFamily } from '../../../state/entityFamily';
import { NodeResizeControl, Handle, Position } from '@xyflow/react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { strategyRequestChildrenSelector } from '../../../state/entitiesSelectors';
import { EntityTypes } from './EntityEnum';

const NodeContent = React.memo(
    ({ localEntity, updateLocalField, handleCreateChild, children }) => {
      return (
        <div className="h-full w-full">
          {children({ entity: localEntity, updateLocalField, handleCreateChild })}
        </div>
      );
    },
    (prevProps, nextProps) => {
      // Ignore position and size changes, but re-render for all other changes
      const prevEntityWithoutPosition = { ...prevProps.localEntity };
      const nextEntityWithoutPosition = { ...nextProps.localEntity };
      
      // Remove position and size properties that change during drag/resize
      delete prevEntityWithoutPosition.position;
      delete nextEntityWithoutPosition.position;
      delete prevEntityWithoutPosition.width;
      delete nextEntityWithoutPosition.width;
      delete prevEntityWithoutPosition.height;
      delete nextEntityWithoutPosition.height;

      const sameEntity = JSON.stringify(prevEntityWithoutPosition) === JSON.stringify(nextEntityWithoutPosition);
      const sameUpdateLocalField = prevProps.updateLocalField === nextProps.updateLocalField;
      const sameHandleCreateChild = prevProps.handleCreateChild === nextProps.handleCreateChild;
      
      // Deep compare the remaining properties
      return sameEntity &&
        sameUpdateLocalField &&
        sameHandleCreateChild;
    }
  );

function EntityNodeBase({ data, children }) {
  // Get the global entity from Recoil (this state is read-only here)
  const [entity] = useRecoilState(entityFamily(data.entityId));
  const { sendStrategyRequest } = useWebSocket();
  // Initialize local state from the global entity only once
  const [localEntity, setLocalEntity] = useState(entity);

  // Memoize the updater so its identity stays the same between renders
  const updateLocalField = useCallback((field, value) => {
    setLocalEntity(prev => ({
      ...prev,
      [field]: value,
    }));
  }, []);

  const updateEntity = useRecoilCallback(({ set }) => (childId, updatedFields) => {
    set(entityFamily(childId), (prev) => ({
      ...prev,
      ...updatedFields,
    }));
    console.log('Updated entity:', childId, updatedFields);
  }, []);

  const strategyRequestChildren = useRecoilValue(strategyRequestChildrenSelector(data.entityId));

  // Update localEntity when the global entity changes
  useEffect(() => {
    setLocalEntity(entity);
  }, [entity]);

  // Use a ref to track if we've already processed these children
  const processedChildrenRef = React.useRef({});
  
  // Update with proper dependency array to re-run when children change
  useEffect(() => {
    if (strategyRequestChildren.length > 0) {
      // Create a batch of updates to avoid multiple re-renders
      const updatesToMake = [];
      
      strategyRequestChildren.forEach((child) => {
        // Only update children we haven't processed yet
        if (!processedChildrenRef.current[child.entity_id]) {
          updatesToMake.push(child);
          // Mark this child as processed
          processedChildrenRef.current[child.entity_id] = true;
        }
      });
      
      // If we have new updates to make, do them all at once
      if (updatesToMake.length > 0) {
        updatesToMake.forEach((child) => {
          updateEntity(child.entity_id, {
            hidden: true,
            width: 500,
          });
        });
      }
    }
  }, [strategyRequestChildren, updateEntity]);

  // Memoize handleCreateChild as well
  const handleCreateChild = useCallback(() => {
    const request = {
      strategy_name: 'CreateEntityStrategy',
      target_entity_id: data.entityId,
      param_config: {
        entity_class: 'shared_utils.entities.StrategyRequestEntity.StrategyRequestEntity',
      },
      add_to_history: true,
      nested_requests: [],
    };
    sendStrategyRequest(request);
  }, [data.entityId, sendStrategyRequest]);

  // Use the size from the global entity for layout
  const { width = 200, height = 100 } = entity;

  return (
    <div
      style={{
        width,
        height,
        position: 'relative',
        backgroundColor: '#1f2937',
        border: '1px solid #374151',
        borderRadius: 4,
        color: 'white',
        overflow: 'hidden',
      }}
    >
      <NodeResizeControl minWidth={100} minHeight={50}>
        <ResizeIcon />
      </NodeResizeControl>

      <Handle type="target" position={Position.Top} />

      {/* The container for node content is wrapped in our memoized NodeContent.
          This prevents the text input from re-rendering on every node position update. */}
      <div className="flex-grow h-full w-full p-4 m-4 flex items-center justify-center">
        <NodeContent
          localEntity={localEntity}
          updateLocalField={updateLocalField}
          handleCreateChild={handleCreateChild}
        >
          {children}
        </NodeContent>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function ResizeIcon() {
  return (
    <div 
      style={{ 
        position: 'absolute', 
        right: 5, 
        bottom: 5,
        padding: '2px',
        backgroundColor: '#1f2937', // Dark gray background
        border: '1px solid #4b5563', // Gray border
        borderRadius: '4px',
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        strokeWidth="2"
        stroke="#9ca3af" // Gray color
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path stroke="none" d="M0 0h24v24H0z" fill="none" />
        <polyline points="16 20 20 20 20 16" />
        <line x1="14" y1="14" x2="20" y2="20" />
        <polyline points="8 4 4 4 4 8" />
        <line x1="4" y1="4" x2="10" y2="10" />
      </svg>
    </div>
  );
}

export default EntityNodeBase;