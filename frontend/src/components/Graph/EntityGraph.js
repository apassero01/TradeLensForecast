import React, { useCallback, useMemo, useRef, useEffect } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Position
} from 'reactflow';
import dagre from 'dagre';
import { processEntityGraph } from '../../utils/graphDataProcessor';
import EntityNode from './EntityNode';

const nodeTypes = {
  entityNode: EntityNode
};

const EntityGraph = ({ data, onNodeClick, selectedEntity }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const nodePositions = useRef(new Map());

  const getLayoutedElements = (nodes, edges, direction = 'TB') => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    
    // Create a map of parent to children
    const parentChildMap = new Map();
    const nodePositions = new Map();
    const nodeHeights = new Map(); // Track node heights
    
    // First, map all parent-child relationships and store node heights
    edges.forEach(edge => {
        if (!parentChildMap.has(edge.source)) {
            parentChildMap.set(edge.source, []);
        }
        parentChildMap.get(edge.source).push(edge.target);
    });

    // Calculate and store node heights based on data
    nodes.forEach(node => {
        // You might need to adjust this calculation based on your EntityNode component
        const baseHeight = 100; // minimum height
        const contentLines = node.data?.metaData ? Object.keys(node.data.metaData).length : 0;
        const estimatedHeight = baseHeight + (contentLines * 20); // 20px per line of content
        nodeHeights.set(node.id, estimatedHeight);
    });
    
    // Find root nodes (nodes with no parents)
    const childNodes = new Set(edges.map(e => e.target));
    const rootNodes = nodes.filter(node => !childNodes.has(node.id)).map(node => node.id);
    
    // Constants for spacing
    const MIN_VERTICAL_SPACING = 25;  // Minimum space between bottom of parent and top of child
    const HORIZONTAL_SPACING = 250;
    
    // Recursive function to position nodes
    const positionNode = (nodeId, parentX = 0, parentY = 0, parentHeight = 0) => {
        const currentNodeHeight = nodeHeights.get(nodeId) || 100;
        
        // If this node already has a position, use that
        if (!nodePositions.has(nodeId)) {
            // For child nodes, account for parent's height and minimum spacing
            const yPosition = parentHeight > 0 
                ? parentY + parentHeight/2 + MIN_VERTICAL_SPACING + currentNodeHeight/2
                : parentY;
            
            nodePositions.set(nodeId, { 
                x: parentX, 
                y: yPosition,
                height: currentNodeHeight
            });
        }
        
        const children = parentChildMap.get(nodeId) || [];
        
        if (children.length > 0) {
            // Calculate total width needed for children
            const totalWidth = (children.length - 1) * HORIZONTAL_SPACING;
            // Calculate starting x position to center children under parent
            const startX = parentX - (totalWidth / 2);
            
            // Position each child
            children.forEach((childId, index) => {
                const childX = startX + (index * HORIZONTAL_SPACING);
                const nodePos = nodePositions.get(nodeId);
                positionNode(
                    childId, 
                    childX, 
                    nodePos.y, 
                    currentNodeHeight
                );
            });
        }
    };
    
    // Position root nodes and their subtrees
    rootNodes.forEach((rootId, index) => {
        const rootX = index * (HORIZONTAL_SPACING * 2); // Space out root nodes
        positionNode(rootId, rootX, 0, 0);
    });
    
    // Apply positions to nodes
    const positionedNodes = nodes.map(node => ({
        ...node,
        position: nodePositions.get(node.id) || { x: 0, y: 0 },
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
    }));

    // Style edges
    const styledEdges = edges.map(edge => ({
        ...edge,
        type: 'smoothstep',
        animated: false,
        style: { strokeWidth: 2, stroke: '#4B5563' },
        markerEnd: {
            type: 'arrowclosed',
            width: 12,
            height: 12,
            color: '#4B5563'
        },
        sourceHandle: 'bottom',
        targetHandle: 'top',
    }));

    return {
        nodes: positionedNodes,
        edges: styledEdges
    };
  };

  // Process the graph data using memoization
  const layoutedElements = useMemo(() => {
    if (!data) return { nodes: [], edges: [] };
    
    const processed = processEntityGraph(data);
    
    const updatedNodes = processed.nodes.map(node => ({
      ...node,
      position: nodePositions.current.get(node.data.path) || node.position,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
    }));

    return getLayoutedElements(updatedNodes, processed.edges, 'TB');
  }, [data]);

  // Update state in useEffect instead of useMemo
  useEffect(() => {
    setNodes(layoutedElements.nodes);
    setEdges(layoutedElements.edges);
  }, [layoutedElements, setNodes, setEdges]);

  const onNodeDragStop = useCallback((event, node) => {
    nodePositions.current.set(node.data.path, node.position);
  }, []);

  const handleNodeClick = useCallback((event, node) => {
    onNodeClick(event, node);
  }, [onNodeClick]);

  const handleKeyDown = useCallback((event) => {
    if (event.key === 'Backspace' || event.key === 'Delete') {
      event.preventDefault();
      event.stopPropagation();
    }
  }, []);

  // Style nodes based on selection
  const nodesWithStyles = useMemo(() => {
    return nodes.map(node => {
      const isSelected = selectedEntity?.id === node.id;
      return {
        ...node,
        data: {
          ...node.data,
          selected: isSelected
        },
        style: {
          backgroundColor: isSelected ? 'rgba(34, 197, 94, 0.1)' : 'transparent',
          borderColor: isSelected ? '#22c55e' : '#4b5563',
          borderWidth: isSelected ? '2px' : '1px',
          boxShadow: isSelected ? '0 0 0 2px rgba(34, 197, 94, 0.5)' : 'none',
        }
      };
    });
  }, [nodes, selectedEntity]);

  if (!data) {
    return (
      <div className="h-screen flex items-center justify-center text-gray-400">
        Loading graph data...
      </div>
    );
  }

  return (
    <div className="h-screen" onKeyDown={handleKeyDown} tabIndex={-1}>
      <ReactFlow
        nodes={nodesWithStyles}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        deleteKeyCode={null}
        fitView
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: true
        }}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
};

export default React.memo(EntityGraph); 