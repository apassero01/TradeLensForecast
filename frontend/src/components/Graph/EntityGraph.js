import React, { useCallback, useMemo, useRef, useEffect } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import EntityNode from './EntityNode';

const nodeTypes = {
  entityNode: EntityNode
};

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: true
};

const EntityGraph = React.memo(({ 
  nodes: initialNodes = [],
  edges: initialEdges = [],
  onNodeClick,
  selectedEntity,
  onNodesChange: onExternalNodesChange 
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    if (initialNodes && initialEdges) {
      const updatedNodes = initialNodes.map(newNode => {
        const existingNode = nodes.find(n => n.id === newNode.id);
        if (existingNode) {
          return {
            ...newNode,
            position: existingNode.position
          };
        }
        return newNode;
      });
      
      setNodes(updatedNodes);
      setEdges(initialEdges);
    }
  }, [initialNodes, initialEdges]);

  const handleNodesChange = useCallback((changes) => {
    const positionChanges = changes.filter(change => 
      change.type === 'position' || change.type === 'dimensions'
    );
    
    if (positionChanges.length > 0) {
      onNodesChange(positionChanges);
    }
    
    if (onExternalNodesChange) {
      onExternalNodesChange(positionChanges);
    }
  }, [onNodesChange, onExternalNodesChange]);

  const styledNodes = useMemo(() => {
    if (!nodes) return [];
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
    <div className="h-screen">
      <ReactFlow
        nodes={styledNodes}
        edges={edges || []}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
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
});

export default EntityGraph; 