// src/components/Canvas/Canvas.jsx
import React, { useMemo, useEffect, useCallback, useState, useRef } from 'react';
import { ReactFlow, Background, BackgroundVariant, applyNodeChanges, Controls, applyEdgeChanges } from '@xyflow/react';
import { useRecoilValue, useRecoilCallback } from 'recoil';
import { flowNodesSelector, flowEdgesSelector, hiddenPropertiesSelector, allEntitiesSelector } from '../../state/entitiesSelectors';
import '@xyflow/react/dist/style.css';
import ContextMenu from './ContextMenu';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
import DynamicNodeWrapper from './Entity/NodeWrapper';
import { entityIdsAtom } from '../../state/entityIdsAtom';
import { entityFamily } from '../../state/entityFamily';
import HiddenSyncer from './HiddenSyncer';

function Canvas() {
  // 1. Read the nodes & edges from Recoil
  const nodeIds = useRecoilValue(entityIdsAtom)
  const allEntities = useRecoilValue(allEntitiesSelector)
  const { sendStrategyRequest } = useWebSocketConsumer();
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [menu, setMenu] = useState(null);
  const reactFlowRef = useRef(null);

  useEffect(() => {
    // For each active node, build a React Flow node using the backend position/dimensions.
    // Here you might fetch the current Recoil state for each node (or have that selector provide you a snapshot).
    const newNodes = nodeIds.map((id) => {
      // For simplicity, assume you can get the current state from Recoil via an external function or selector.
      // Alternatively, you might store a local mapping of id -> node properties.
      // In this example, we assume a default if no state is available.
      // IMPORTANT: You want to avoid recreating nodes on every render; ideally, this runs only on add/remove or backend update.
      return {
        id,
        type: 'dynamic',
        // Use the backend-specified position/dimensions when available.
        // These values might come from Recoil or a dedicated selector.
        position: { x: 0, y: 0 }, // Replace with actual backend position if available
        data: { entityId: id },
        hidden: false,
      };
    });
    setNodes(newNodes);
  }, [nodeIds]); // or add dependency on backend state changes
  
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
      "dynamic": DynamicNodeWrapper,
    }
  }, []);


  const onEdgesChange = useCallback((changes) => {
    console.log('Edges changed', changes);
    setEdges((prevEdges) => applyEdgeChanges(changes, prevEdges));
  }, []);

  const onNodesChange = useCallback((changes) => {
    // Update nodes based on the changes
    console.log('Nodes changed', changes);
    setNodes((prevNodes) => {
      // Use applyNodeChanges to handle updates efficiently
      const updatedNodes = applyNodeChanges(changes, prevNodes);
      
      // Handle dimension changes inside the setter function where we have the latest nodes
      changes.forEach((change) => {
        if (change.type === 'dimensions' && change.resizing === false) {
          const entity = updatedNodes.find((node) => node.id === change.id);
          if (entity) {
            console.log('Updating entity', entity);
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
        if (change.type === 'position' && !change.dragging) {
          const entity = updatedNodes.find((node) => node.id === change.id);
          if (entity) {
            console.log('Updating entity', entity);
            sendStrategyRequest({
              strategy_name: 'SetAttributesStrategy',
              param_config: {
                attribute_map: {
                  'position': {
                    'x': entity.position.x,
                    'y': entity.position.y
                  }
                }
              },
              target_entity_id: entity.data.entityId,
              add_to_history: false,
              nested_requests: [],
            })
          }
        }
      });
      
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

  const onConnect = useCallback((connection) => {
    // Create new edge
    setEdges((prevEdges) => [...prevEdges, connection]);

    // Send strategy request where source is assigner and target is assignee
    sendStrategyRequest({
      strategy_name: 'AddChildStrategy',
      param_config: {
        child_id: connection.target, // The node being assigned to (target/child)
      },
      target_entity_id: connection.source, // The node doing the assigning (source/parent)
      add_to_history: false,
      nested_requests: [],
    });
  }, [sendStrategyRequest]);

  const onInit = useCallback((instance) => {
    reactFlowRef.current = instance;
  }, []);

  return (
    <div style={{ width: '100%', height: '100vh', flex: 1 }}>
      <ReactFlow
        ref={reactFlowRef}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onMove={onMoveEnd}
        onEdgesChange={onEdgesChange}
        fitView
        onNodeContextMenu={onNodeContextMenu}
        onPaneClick={onPaneClick}
        onConnect={onConnect}
        selectionOnDrag={true}
        selectionMode={"partial"}
        panOnDrag={true}
        snapToGrid={false}
        maxZoom={10}
        minZoom={0.1}
        onInit={onInit}
      >
        <HiddenSyncer />
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