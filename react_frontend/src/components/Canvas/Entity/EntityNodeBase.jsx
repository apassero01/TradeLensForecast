import React, { useCallback, useEffect, useRef } from 'react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { NodeResizeControl, Handle, Position } from '@xyflow/react';
import { useWebSocketConsumer } from '../../../hooks/useWebSocketConsumer';
import { strategyRequestChildrenSelector, childrenByTypeSelector } from '../../../state/entitiesSelectors';
import StrategyRequestList from '../../Strategy/StrategyRequestList';
import { FaPlus } from 'react-icons/fa';
import { StrategyRequests } from '../../../utils/StrategyRequestBuilder';
function EntityNodeBase({ data, children, updateEntity }) {
  // const entity = useRecoilValue(entityFamily(data.entityId)); // Use full entity from Recoil
  const { sendStrategyRequest } = useWebSocketConsumer();
  const strategyRequestChildren = useRecoilValue(strategyRequestChildrenSelector(data.entityId));
  // const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: data.entityId, type: EntityTypes.VIEW }));
  const [isLoading, setIsLoading] = React.useState(false);

  // Process strategy request children once per child
  const processedChildrenRef = useRef({});
  useEffect(() => {
    if (strategyRequestChildren?.length > 0) {
      const updatesToMake = [];
      strategyRequestChildren.forEach((child) => {
        if (!processedChildrenRef.current[child.entity_id]) {
          updatesToMake.push(child);
          processedChildrenRef.current[child.entity_id] = true;
        }
      });
      if (updatesToMake.length > 0 && updateEntity) {
        updatesToMake.forEach((child) => {
          sendStrategyRequest(StrategyRequests.hideEntity(child.entity_id, true));
          updateEntity(child.entity_id, {
            hidden: true,
          });
        });
      }
    }

    // if (viewChildren) {
    //   viewChildren.forEach((child) => {
    //     if (!processedChildrenRef.current[child.entity_id]) {
    //       updateEntity(child.entity_id, { hidden: child?.hidden });
    //       processedChildrenRef.current[child.entity_id] = true;
    //     }
    //   });
    // }
  }, [strategyRequestChildren, updateEntity]);

  // Create a child strategy request
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

  // Remove a child
  const handleRemoveChild = useCallback(
    (child_id) => {
      sendStrategyRequest({
        strategy_name: 'RemoveChildStrategy',
        target_entity_id: data.entityId,
        param_config: { child_id },
        add_to_history: false,
        nested_requests: [],
      });
    },
    [data.entityId, sendStrategyRequest]
  );

  const handleNodeClick = useCallback((event) => {
    // Prevent event propagation to avoid canvas deselection
    event.stopPropagation();
    
    if (updateEntity) {
      // Toggle selected state and update zIndex
      updateEntity(data.entityId, {
        selected: !data.selected,
        zIndex: !data.selected ? 999 : 0, // High zIndex when selected, reset when deselected
      });
    }
  }, [data.entityId, data.selected, updateEntity]);

  return (
    <div
      onClick={handleNodeClick}
      style={{
        backgroundColor: '#1f2937',
        border: data.selected 
          ? '2px solid #3b82f6' // Bright blue highlight for selected nodes
          : '1px solid #374151',
        borderRadius: 4,
        color: 'white',
        width: '100%',
        height: '100%',
        position: 'relative',
        boxShadow: data.selected 
          ? '0 0 0 2px rgba(59, 130, 246, 0.5)' // Add a subtle glow for selected nodes
          : 'none',
      }}
      className={isLoading ? 'entity-loading' : ''}
    >
      {isLoading && (
        <div className="absolute inset-0 rounded-[3px] z-10">
          <div className="absolute inset-0 rounded-[3px] border-2 border-blue-500 animate-pulse"></div>
        </div>
      )}
      
      <NodeResizeControl minWidth={100} minHeight={50}>
        <ResizeIcon />
      </NodeResizeControl>

      <Handle type="target" position={Position.Top} />

      <div className="flex-grow h-full w-full p-4 flex items-center justify-center overflow-y-hidden overflow-x-hidden">
        {children({
          data: data,
          childrenRequests: strategyRequestChildren,
          updateEntity: updateEntity,
          sendStrategyRequest: sendStrategyRequest,
          onRemoveRequest: handleRemoveChild,
          isLoading: isLoading,
          setIsLoading: setIsLoading,
        })}
      </div>

      <button
        onClick={handleCreateChild}
        style={{
          position: 'absolute',
          top: '8px',
          right: '8px',
          backgroundColor: '#3b82f6', // Bright blue from your description
          color: 'white',
          border: 'none',
          borderRadius: '50%',
          width: '24px',
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
      >
        <FaPlus size={12} /> {/* Assuming FaPlus from FontAwesome */}
      </button>

      <Handle type="source" position={Position.Bottom} />

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
        <StrategyRequestList
          childrenRequests={strategyRequestChildren}
          updateEntity={updateEntity}
          sendStrategyRequest={sendStrategyRequest}
          onRemoveRequest={handleRemoveChild}
          data={data}
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
        backgroundColor: '#374151',
        borderRadius: '2px',
        cursor: 'nwse-resize',
      }}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
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