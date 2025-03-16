// src/components/Canvas/Canvas.jsx
import React, { useMemo, useEffect, useCallback, useState, useRef } from 'react';
import { ReactFlow, Background, BackgroundVariant, applyNodeChanges, Controls } from '@xyflow/react';
import { useRecoilValue } from 'recoil';
import { flowNodesSelector, flowEdgesSelector } from '../../state/entitiesSelectors';
import EntityNode from './Entity/EntityNode';
import StrategyRequestEntity from './Entity/StrategyRequestEntity';
import InputEntity from './Entity/InputEntity';
import VisualizationEntity from './Entity/VisualizationEntity/VisualizationEntity';
import '@xyflow/react/dist/style.css';
import ContextMenu from './ContextMenu';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
function Canvas() {
  // 1. Read the nodes & edges from Recoil
  const recoilNodes = useRecoilValue(flowNodesSelector);
  const recoilEdges = useRecoilValue(flowEdgesSelector);
  const { sendStrategyRequest } = useWebSocketConsumer();
  const [nodes, setNodes] = useState(recoilNodes);
  const [menu, setMenu] = useState(null);
  const reactFlowRef = useRef(null);
  
  // Instead of using the hook at component level, track mouse position directly
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  
  // Keep track of current zoom level
  const [zoom, setZoom] = useState(1);
  
  // Track mouse position
  useEffect(() => {
    const handleMouseMove = (event) => {
      setMousePosition({
        x: event.clientX,
        y: event.clientY
      });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  // Update zoom level when ReactFlow reports changes
  const onMoveEnd = useCallback((event) => {
    // Make sure event exists and has a defined zoom property
    if (event && typeof event.zoom === 'number') {
      setZoom(event.zoom);
    }
  }, []);

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

  const onNodeContextMenu = useCallback(
    (event, node) => {
      // Prevent native context menu from showing
      event.preventDefault();
      
      // Set menu position directly at cursor position
      setMenu({
        id: node.id,
        entityId: node.data.entityId,
        // Position menu with top-left corner at cursor position
        top: event.clientY,
        left: event.clientX,
        // Clear any potential right/bottom values
        right: undefined,
        bottom: undefined,
      });
    },
    []
  );

  // Close the context menu when clicking elsewhere
  const onPaneClick = useCallback(() => setMenu(null), [setMenu]);

  return (
    <div style={{ width: '100%', height: '100vh', flex: 1 }}>
      <ReactFlow
        ref={reactFlowRef}
        nodes={nodes}
        edges={recoilEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onMove={onMoveEnd}
        fitView
        onNodeDragStop={(event, node) => {
          onDragEnd(event, node);
        }}
        onNodeContextMenu={onNodeContextMenu}
        onPaneClick={onPaneClick}
      >
        <Background
          id="2"
          gap={25}
          size={0.25}
          color="#ccc"
          variant={BackgroundVariant.Cross}
        />
        <Controls />
        {menu && <ContextMenu onClick={onPaneClick} {...menu} />}
      </ReactFlow>
    </div>
  );
}

export default Canvas;