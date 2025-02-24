// src/components/Canvas/Canvas.jsx
import React, { useMemo } from 'react';
import { ReactFlow, Background, BackgroundVariant } from '@xyflow/react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { flowNodesSelector, flowEdgesSelector } from '../../state/entitiesSelectors';
import { entityFamily } from '../../state/entityFamily';
import EntityNode from './EntityNode';
import StrategyRequestEntity from './StrategyRequestEntity';
import '@xyflow/react/dist/style.css';

function Canvas() {
  // 1. Read the nodes & edges from Recoil
  const nodes = useRecoilValue(flowNodesSelector);
  const edges = useRecoilValue(flowEdgesSelector);

  const onNodesChange = useRecoilCallback(
    ({ set }) =>
      (changes) => {
        changes.forEach((change) => {
          // Filter out 'select' or other event types we don't need
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

  // 3. Optionally augment each node's data
  const nodesWithData = useMemo(() => {
    return nodes.map((node) => ({
      ...node,
      position: node.position || { x: 0, y: 0 },
      data: {
        ...node.data,
        entityId: node.id, // So EntityNode can read from Recoil
      },
    }));
  }, [nodes]);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodesWithData}
        edges={edges}
        nodeTypes={{ 
          entityNode: EntityNode, 
          strategyRequestEntity: StrategyRequestEntity 
        }}
        onNodesChange={onNodesChange}
        fitView
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