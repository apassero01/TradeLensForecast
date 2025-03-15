// src/components/Canvas/Canvas.jsx
import React, { useMemo, useEffect, useCallback, useState } from 'react';
import { ReactFlow, Background, BackgroundVariant, applyNodeChanges, Controls } from '@xyflow/react';
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
  const recoilNodes = useRecoilValue(flowNodesSelector);
  const recoilEdges = useRecoilValue(flowEdgesSelector);
  const { sendStrategyRequest } = useWebSocket();
  const [nodes, setNodes] = useState(recoilNodes);

  const nodeTypes = useMemo(() => {
    return {
      entityNode: EntityNode, 
      strategyRequestEntity: StrategyRequestEntity,
      inputEntity: InputEntity,
      visualizationEntity: VisualizationEntity,
    }
  }, []);


  useEffect(() => {
    setNodes((prevNodes) => 
      recoilNodes.map((recoilNode) => {
        const existingNode = prevNodes.find((n) => n.id === recoilNode.id);
        return {
          ...recoilNode,
          position: existingNode?.position || recoilNode.position,
          width: existingNode?.width || recoilNode.width,
          height: existingNode?.height || recoilNode.height,
        }
      })
    )
  }, [recoilNodes]);


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

  const onNodesChange = useCallback((changes) => {
    // Update nodes based on the changes
    setNodes((prevNodes) => {
      // Use applyNodeChanges to handle updates efficiently
      const updatedNodes = applyNodeChanges(changes, prevNodes);
      
      // Handle dimension changes inside the setter function where we have the latest nodes
      if (changes.length > 0 && changes[0].type === 'dimensions' && changes[0].resizing === false) {
        const entity = updatedNodes.find((node) => node.id === changes[0].id);
        if (entity) {
          sendStrategyRequest({
            strategy_name: 'SetAttributesStrategy',
            param_config: {
              attribute_map: {
                'width': entity.width,
                'height': entity.height,
              }
            },
            target_entity_id: entity.data.entityId,
            add_to_history: false,
            nested_requests: [],
          });
        }
      }
      
      return updatedNodes;
    });
  }, [sendStrategyRequest]);

  return (
    <div style={{ width: '100%', height: '100vh', flex: 1 }}>
      <ReactFlow
        nodes={nodes}
        edges={recoilEdges}
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
        <Controls />
      </ReactFlow>
    </div>
  );
}

export default Canvas;