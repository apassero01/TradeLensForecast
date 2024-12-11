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

      // Preserve position for existing nodes
      if (existingNode) {
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

    const newNodes = processNodeUpdates(processedElements.nodes, processedElements.edges);
    prevNodesRef.current = new Map(newNodes.map(node => [node.data.path, node]));
    
    setNodes(newNodes);
    setEdges(processedElements.edges);
  }, [processedElements, processNodeUpdates, setNodes, setEdges]);

  const onNodeDragStop = useCallback((event, node) => {
    nodePositions.current.set(node.data.path, node.position);
  }, []);

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