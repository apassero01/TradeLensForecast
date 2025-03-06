// src/components/Canvas/EntityNodeBase.jsx
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { entityFamily } from '../../../state/entityFamily';
import { NodeResizeControl, Handle, Position } from '@xyflow/react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { strategyRequestChildrenSelector, nonTransientEntitySelector } from '../../../state/entitiesSelectors';
import StrategyRequestList from '../../Strategy/StrategyRequestList';
import { FaPlus } from 'react-icons/fa';

// Memoized content for the node, ignoring transient properties
const MemoizedNodeContent = React.memo(
  ({ entity, updateLocalField, handleCreateChild, updateEntity, sendStrategyRequest, children }) => {
    return (
      <div className="h-full w-full">
        {children({
          entity,
          updateLocalField,
          handleCreateChild,
          updateEntity,
          sendStrategyRequest,
        })}
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Create shallow copies to remove transient properties (if any)
    const prevEntity = { ...prevProps.entity };
    const nextEntity = { ...nextProps.entity };

    const sameEntity = JSON.stringify(prevEntity) === JSON.stringify(nextEntity);
    const sameUpdateLocalField = prevProps.updateLocalField === nextProps.updateLocalField;
    const sameHandleCreateChild = prevProps.handleCreateChild === nextProps.handleCreateChild;

    return sameEntity && sameUpdateLocalField && sameHandleCreateChild;
  }
);

// Memoize the StrategyRequestList separately so that it only re-renders when its props change.
const MemoizedStrategyRequestList = React.memo(StrategyRequestList);

function EntityNodeBase({ data, children }) {
  // Subscribe only to the stable (non-transient) properties of the entity
  const stableEntity = useRecoilValue(nonTransientEntitySelector(data.entityId));
  const { sendStrategyRequest } = useWebSocket();

  // console.log('stableEntity', stableEntity);

  // Separate visual state for transient properties (position, width, height)
  const [visualState, setVisualState] = useState({
    width: stableEntity.width || 200,
    height: stableEntity.height || 100,
  });

  // Keep local state for the stable data if needed; here we directly use stableEntity

  // Memoize updater for visual state changes
  const updateLocalField = useCallback((field, value) => {
    setVisualState((prev) => ({
      ...prev,
      [field]: value,
    }));
  }, []);

  // Callback to update an entity atom (or its child)
  const updateEntity = useRecoilCallback(({ set }) => (childId, updatedFields) => {
    set(entityFamily(childId), (prev) => ({
      ...prev,
      ...updatedFields,
    }));
    console.log('Updated entity:', childId, updatedFields);
  }, []);

  // Get the strategy request children from recoil
  const strategyRequestChildren = useRecoilValue(strategyRequestChildrenSelector(data.entityId));

  // Process strategy request children only once per child using a ref
  const processedChildrenRef = useRef({});
  useEffect(() => {
    if (strategyRequestChildren.length > 0) {
      const updatesToMake = [];
      strategyRequestChildren.forEach((child) => {
        if (!processedChildrenRef.current[child.entity_id]) {
          updatesToMake.push(child);
          processedChildrenRef.current[child.entity_id] = true;
        }
      });
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

  // Memoized callback for creating a child strategy request
  const handleCreateChild = useCallback(() => {
    const request = {
      strategy_name: 'CreateEntityStrategy',
      target_entity_id: data.entityId,
      param_config: {
        entity_class: 'shared_utils.entities.StrategyRequestEntity.StrategyRequestEntity',
      },
      add_to_history: false,
      nested_requests: [],
    };
    sendStrategyRequest(request);
  }, [data.entityId, sendStrategyRequest]);

  // Memoized callback for removing a child
  const handleRemoveChild = useCallback((child_id) => {
    sendStrategyRequest({
      strategy_name: 'RemoveChildStrategy',
      target_entity_id: data.entityId,
      param_config: { child_id },
      add_to_history: false,
      nested_requests: [],
    });
  }, [data.entityId, sendStrategyRequest]);

  const { width, height } = visualState;

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
        overflow: 'visible', // Allow overflow for the strategy list
      }}
    >
      <NodeResizeControl minWidth={100} minHeight={50}>
        <ResizeIcon />
      </NodeResizeControl>

      <Handle type="target" position={Position.Top} />

      {/* Use memoized node content for stable data */}
      <div
        className="flex-grow h-full w-full p-4 flex items-center justify-center"
        style={{ paddingBottom: strategyRequestChildren.length > 0 ? '40px' : '4px' }}
      >
        <MemoizedNodeContent
          entity={stableEntity}
          updateLocalField={updateLocalField}
          handleCreateChild={handleCreateChild}
          updateEntity={updateEntity}
          sendStrategyRequest={sendStrategyRequest}
        >
          {children}
        </MemoizedNodeContent>
      </div>

      {/* Create Strategy Button */}
      <button
        onClick={handleCreateChild}
        style={{
          position: 'absolute',
          bottom: '8px',
          right: '8px',
          backgroundColor: '#3b82f6',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          padding: '4px 8px',
          display: 'flex',
          alignItems: 'center',
          fontSize: '12px',
          cursor: 'pointer',
          zIndex: 10,
        }}
        title="Create Strategy Request"
      >
        <FaPlus size={10} style={{ marginRight: '4px' }} />
      </button>

      <Handle type="source" position={Position.Bottom} />

      {/* Memoized Strategy Request List positioned to the right */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: '100%',
          height: '100%',
          zIndex: 100,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <MemoizedStrategyRequestList
          childrenRequests={strategyRequestChildren}
          updateEntity={updateEntity}
          sendStrategyRequest={sendStrategyRequest}
          onRemoveRequest={handleRemoveChild}
        />
      </div>
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
        backgroundColor: '#1f2937',
        border: '1px solid #4b5563',
        borderRadius: '4px',
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="20"
        height="20"
        viewBox="0 0 24 24"
        strokeWidth="2"
        stroke="#9ca3af"
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

export default React.memo(EntityNodeBase);