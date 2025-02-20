import React, { useCallback, useMemo, useEffect } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import EntityNode from '../Entity/EntityNode';
import StrategyRequest from '../../utils/StrategyRequest';
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
  onNodesChange: onExternalNodesChange,
  onStrategyExecute,
  onStrategyListExecute,
  // This is newly passed from EntityGraphApp
  availableStrategies,
  fetchAvailableStrategies
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

  const handleNodeDragStop = useCallback((event, node) => {
    console.log('Node dragged:', node);
    const strategyRequest = new StrategyRequest({
      name: "SetAttributesStrategy",
      config: {
        attribute_map: {
          "position": node.position
        }
      },
      target_entity_id: node.id,
      add_to_history: false,
    })
    onStrategyExecute(strategyRequest);
  }, []);

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

  // Inject availableStrategies into each node's data
  const styledNodes = useMemo(() => {
    if (!nodes) return [];
    return nodes.map(node => ({
      ...node,
      data: {
        ...node.data,
        onStrategyExecute,
        onStrategyListExecute,
        // ADDED:
        availableStrategies: availableStrategies,
        getAvailableStrategies: fetchAvailableStrategies
      },
      style: {
        ...node.style,
        borderColor: selectedEntity?.id === node.id ? '#3b82f6' : undefined,
        borderWidth: selectedEntity?.id === node.id ? 2 : undefined,
      },
    }));
  }, [
    nodes, 
    selectedEntity, 
    onStrategyExecute, 
    onStrategyListExecute, 
    availableStrategies
  ]);

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
        onNodeDragStop={handleNodeDragStop}
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
});

export default EntityGraph;