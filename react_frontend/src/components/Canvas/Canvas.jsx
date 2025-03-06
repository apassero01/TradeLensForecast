// src/components/Canvas/Canvas.jsx
import React, { useMemo, useEffect, useCallback } from 'react';
import { ReactFlow, Background, BackgroundVariant } from '@xyflow/react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { flowNodesSelector, flowEdgesSelector } from '../../state/entitiesSelectors';
import { entityFamily } from '../../state/entityFamily';
import EntityNode from './Entity/EntityNode';
import StrategyRequestEntity from './Entity/StrategyRequestEntity';
import InputEntity from './Entity/InputEntity';
import VisualizationEntity from './Entity/VisualizationEntity/VisualizationEntity';
import '@xyflow/react/dist/style.css';
import { useWebSocket } from '../../hooks/useWebSocket';



function Canvas() {
  // 1. Read the nodes & edges from Recoil
  const nodes = useRecoilValue(flowNodesSelector);
  const edges = useRecoilValue(flowEdgesSelector);
  const { sendStrategyRequest } = useWebSocket();

  const nodeTypes = useMemo(() => {
    return {
      entityNode: EntityNode, 
      strategyRequestEntity: StrategyRequestEntity,
      inputEntity: InputEntity,
      visualizationEntity: VisualizationEntity
    }
  }, []);

  const onDragEnd = useCallback((event, node) => {
    console.log('onDragEnd', event, node);
    if (!node.position.x || !node.position.y) {
      return;
    }
    sendStrategyRequest({
      strategy_name: 'SetAttributesStrategy',
      param_config: {
        attribute_map: {
          'position': {
            'x': node.position.x,
            'y': node.position.y
          }
        }
      },
      target_entity_id: node.data.entityId,
      add_to_history: false,
      nested_requests: [],
    })
  }, [sendStrategyRequest]);

  const onNodesChange = useRecoilCallback(
    ({ set }) =>
      (changes) => {
        changes.forEach((change) => {
          // Filter out 'select' or other event types we don't need
          const request = {
            strategy_name: 'SetAttributesStrategy',
            param_config: {
              attribute_map: {
              }
            }
          }
          if (change.type === 'select') {
            return;
          }
  
          // 1. Dimensions changes (resizing)
          if (change.type === 'dimensions' && change.dimensions) {
            const { id, dimensions } = change;
            const { width: newWidth, height: newHeight } = dimensions;
  
            // If newWidth or newHeight is NaN, skip
            if (
              Number.isNaN(newWidth) ||
              Number.isNaN(newHeight)
            ) {
              return;
            }
  
            set(entityFamily(id), (prev) => {
              // Compare old vs. new
              if (prev.width === newWidth && prev.height === newHeight) {
                return prev; // No change
              }
              return {
                ...prev,
                width: newWidth,
                height: newHeight,
              };
            });
          }

          // 2. Position changes (dragging/moving)
          else if (change.type === 'position' && change.position) {
            const { id, position } = change;
            const { x: newX, y: newY } = position;
  
            // If newX or newY is NaN, skip
            if (
              Number.isNaN(newX) ||
              Number.isNaN(newY)
            ) {
              return;
            }
  
            set(entityFamily(id), (prev) => {
              // Compare old vs. new
              if (
                prev.position?.x === newX &&
                prev.position?.y === newY
              ) {
                return prev; // No change
              }
              return {
                ...prev,
                position: { x: newX, y: newY },
              };
            });
          }
        });
      },
    []
  );

  // // 3. Optionally augment each node's data
  // const nodesWithData = useMemo(() => {
  //   return nodes.map((node) => ({
  //     ...node,
  //     position: node.position || { x: 0, y: 0 },
  //     data: {
  //       ...node.data,
  //       entityId: node.id, // So EntityNode can read from Recoil
  //     },
  //   }));
  // }, [nodes]);

  return (
    <div style={{ width: '100%', height: '100vh', flex: 1 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes = {nodeTypes}
        onNodesChange={onNodesChange}
        fitView
        onNodeDragStop={(event, node) => {
          // Call your backend update with node position here.
          onDragEnd(event, node);
        }}
      >
        <Background
          id="2"
          gap={25}
          size={0.25}
          color="#ccc"
          variant={BackgroundVariant.Cross}
        />
      </ReactFlow>
    </div>
  );
}

export default Canvas;