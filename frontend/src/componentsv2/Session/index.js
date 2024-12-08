import React, { useState } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  useNodesState, 
  useEdgesState,
} from 'reactflow';
import StrategyPanel from './StrategyPanel';
import 'reactflow/dist/style.css';

const Session = () => {
  const [selectedEntity, setSelectedEntity] = useState(null);

  // Mock available strategies per entity type
  const availableStrategies = {
    'TRAINING_SESSION': [
      'CreateFeatureSet',
      'TrainModel',
      'ExportModel'
    ],
    'FEATURE_SET': [
      'Normalize',
      'HandleMissingValues',
      'FeatureSelection',
      'CreateTimeSeries'
    ],
    'MODEL_STAGE': [
      'Evaluate',
      'Tune',
      'Deploy',
      'SaveCheckpoint'
    ],
    'DATA_BUNDLE': [
      'Clean',
      'Transform',
      'Split',
      'Augment'
    ]
  };

  // Simple nodes representing entities
  const initialNodes = [
    {
      id: '1',
      position: { x: 250, y: 0 },
      data: { label: 'TRAINING_SESSION' },
      style: { 
        background: '#3b82f6',
        color: 'white',
        border: '1px solid #2563eb',
        padding: 10,
        borderRadius: 5,
        cursor: 'pointer'
      }
    },
    {
      id: '2',
      position: { x: 100, y: 100 },
      data: { label: 'FEATURE_SET' },
      style: { 
        background: '#10b981',
        color: 'white',
        border: '1px solid #059669',
        padding: 10,
        borderRadius: 5
      }
    },
    {
      id: '3',
      position: { x: 400, y: 100 },
      data: { label: 'MODEL_STAGE' },
      style: { 
        background: '#8b5cf6',
        color: 'white',
        border: '1px solid #7c3aed',
        padding: 10,
        borderRadius: 5
      }
    },
    {
      id: '4',
      position: { x: 100, y: 200 },
      data: { label: 'DATA_BUNDLE' },
      style: { 
        background: '#f59e0b',
        color: 'white',
        border: '1px solid #d97706',
        padding: 10,
        borderRadius: 5
      }
    }
  ];

  // Simple edges showing relationships
  const initialEdges = [
    { id: 'e1-2', source: '1', target: '2', animated: true },
    { id: 'e1-3', source: '1', target: '3', animated: true },
    { id: 'e2-4', source: '2', target: '4', animated: true }
  ];

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const handleNodeClick = (_, node) => {
    setSelectedEntity(node);
  };

  const handleExecuteStrategy = async (entity, strategy, config) => {
    console.log('Executing strategy:', {
      entityType: entity.data.label,
      strategy,
      config
    });

    try {
      // Format the request according to StrategyRequestEntity structure
      const strategyRequest = {
        strategy_name: strategy,
        param_config: config.param_config || {},
        nested_requests: config.nested_requests || [],
        entity_type: entity.data.label
      };

      // Here you would make the API call
      // const response = await fetch('/api/execute-strategy', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //   },
      //   body: JSON.stringify(strategyRequest)
      // });
      
      // Mock success for now
      console.log('Strategy request:', strategyRequest);
      
      // Update UI to show success
      // You might want to update the node appearance or add a success indicator
      const updatedNodes = nodes.map(node => {
        if (node.id === entity.id) {
          return {
            ...node,
            data: {
              ...node.data,
              lastStrategy: strategy
            },
            // Optionally add a visual indicator of success
            style: {
              ...node.style,
              boxShadow: '0 0 0 2px #10B981'
            }
          };
        }
        return node;
      });
      
      setNodes(updatedNodes);

    } catch (error) {
      console.error('Strategy execution failed:', error);
      // Handle error (show notification, update UI, etc.)
    }
  };

  return (
    <div className="h-screen bg-gray-900 flex">
      <div className="flex-grow">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          fitView
        >
          <Background />
          <Controls />
        </ReactFlow>
      </div>
      
      <StrategyPanel 
        selectedEntity={selectedEntity}
        availableStrategies={availableStrategies}
        onExecuteStrategy={handleExecuteStrategy}
      />
    </div>
  );
};

export default Session; 