import React, { useCallback, useMemo, useRef, useEffect } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Position
} from 'reactflow';
import { processEntityGraph } from '../../utils/graphDataProcessor';
import { calculateNewNodePosition } from '../../utils/nodePositionCalculator';
import EntityNode from './EntityNode';
import { isEqual } from 'lodash';

const nodeTypes = {
  entityNode: EntityNode
};

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: true
};

const EntityGraph = ({ data, onNodeClick, selectedEntity }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const nodePositions = useRef(new Map());
  const prevNodesRef = useRef(new Map());

  // Memoize processed graph data
  const processedElements = useMemo(() => {
    if (!data) return { nodes: [], edges: [] };
    return processEntityGraph(data);
  }, [data]);

  // Find parent and siblings for a node
  const findNodeRelatives = useCallback((nodeId, processedNodes, processedEdges) => {
    const parentEdge = processedEdges.find(edge => edge.target === nodeId);
    if (!parentEdge) return { parentNode: null, siblings: [] };

    const parentNode = prevNodesRef.current.get(
      processedNodes.find(n => n.id === parentEdge.source)?.data.path
    );

    const siblings = processedEdges
      .filter(edge => edge.source === parentEdge.source && edge.target !== nodeId)
      .map(edge => prevNodesRef.current.get(
        processedNodes.find(n => n.id === edge.target)?.data.path
      ))
      .filter(Boolean);

    return { parentNode, siblings };
  }, []);

  // Process node updates while preserving positions
  const processNodeUpdates = useCallback((processedNodes, processedEdges) => {
    return processedNodes.map(node => {
      const existingNode = prevNodesRef.current.get(node.data.path);
      const existingPosition = nodePositions.current.get(node.data.path);

      // If node exists, check for data changes
      if (existingNode) {
        const hasDataChanged = !isEqual(existingNode.data, node.data);
        
        // If no data changes, return existing node with its position
        if (!hasDataChanged) {
          console.log("existingNode Data has not changed");
          return existingNode;
        }
        console.log("existingNode Data has changed");
        // If data changed, return new node with preserved position
        return {
          ...node,
          position: existingPosition || existingNode.position,
          sourcePosition: Position.Bottom,
          targetPosition: Position.Top,
        };
      }

      // Calculate position for new nodes
      const { parentNode, siblings } = findNodeRelatives(node.id, processedNodes, processedEdges);
      const position = calculateNewNodePosition(parentNode, siblings);

      return {
        ...node,
        position,
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
      };
    });
  }, [findNodeRelatives]);

  // Update graph when data changes
  useEffect(() => {
    if (!processedElements.nodes.length) return;

    // Process new nodes while preserving existing positions
    const { nodes: newNodes, edges: newEdges } = processEntityGraph(data, nodes);

    // Only update if data has changed (excluding positions)
    const hasDataChanged = newNodes.some((newNode, index) => {
      const existingNode = nodes[index];
      return !existingNode || 
        !isEqual(
          { ...existingNode.data, position: undefined }, 
          { ...newNode.data, position: undefined }
        );
    });

    if (hasDataChanged) {
      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [data, nodes, setNodes, setEdges]);

  const onNodeDragStop = useCallback((event, node) => {
    // Update only the position
    setNodes(nds => 
      nds.map(n => {
        if (n.id === node.id) {
          return { ...n, position: node.position };
        }
        return n;
      })
    );
  }, [setNodes]);

  const nodesWithStyles = useMemo(() => {
    return nodes.map(node => ({
      ...node,
      style: {
        ...node.style,
        borderColor: selectedEntity?.id === node.id ? '#3b82f6' : undefined,
        borderWidth: selectedEntity?.id === node.id ? 2 : undefined,
      },
    }));
  }, [nodes, selectedEntity]);

  return (
    <div className="h-screen" tabIndex={-1}>
      <ReactFlow
        nodes={nodesWithStyles}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        deleteKeyCode={null}
        fitView
        defaultEdgeOptions={defaultEdgeOptions}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
};

export default React.memo(EntityGraph); 