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

  // Create entity map for O(1) lookups
  const entityMap = useMemo(() => {
    const map = new Map();
    allEntities.forEach(entity => {
      map.set(entity.id, entity);
    });
    return map;
  }, [allEntities]);

  useEffect(() => {
    setNodes((currentNodes) => {
      // Create a map of existing nodes for quick lookup
      const existingNodesMap = new Map(currentNodes.map(node => [node.id, node]));
      
      // Find nodes to add (in nodeIds but not in current nodes)
      const nodesToAdd = nodeIds.filter(id => !existingNodesMap.has(id));
      
      // Find nodes to remove (in current nodes but not in nodeIds)
      const nodeIdsSet = new Set(nodeIds);
      const filteredNodes = currentNodes.filter(node => nodeIdsSet.has(node.id));
      
      // Add new nodes
      const newNodes = nodesToAdd.map((id) => {
        // Try to get position from entity state if available
        const entity = entityMap.get(id);
        const position = entity?.position || { x: Math.random() * 500, y: Math.random() * 500 };
        
        return {
          id,
          type: 'dynamic',
          position,
          data: { entityId: id },
          hidden: false,
          width: entity?.width,
          height: entity?.height,
        };
      });
      
      // Combine filtered existing nodes with new nodes
      return [...filteredNodes, ...newNodes];
    });
  }, [nodeIds, entityMap]);
  
  // Manage edges based on entity relationships
  useEffect(() => {
    const newEdges = [];
    
    // Generate edges based on parent_ids in entity data
    allEntities.forEach(entity => {
      if (entity.data?.parent_ids && Array.isArray(entity.data.parent_ids)) {
        entity.data.parent_ids.forEach(parentId => {
          // Only create edge if both nodes exist
          if (nodeIds.includes(parentId) && nodeIds.includes(entity.id)) {
            newEdges.push({
              id: `${parentId}-${entity.id}`,
              source: parentId,
              target: entity.id,
            });
          }
        });
      }
    });
    
    // Update edges state with the new relationships
    setEdges(newEdges);
  }, [allEntities, nodeIds]);
  
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

  const onEdgesDelete = useCallback((edges) => {
    console.log('Edges deleted', edges);
    sendStrategyRequest({
      strategy_name: 'RemoveChildStrategy',
      param_config: {
        child_id: edges[0].target,
      },
      target_entity_id: edges[0].source,
      add_to_history: false,
      nested_requests: [],
    });
  }, [sendStrategyRequest]);

  const onInit = useCallback((instance) => {
    reactFlowRef.current = instance;
  }, []);

  return (
    <div className="w-full h-full">
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
        onEdgesDelete={onEdgesDelete}
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