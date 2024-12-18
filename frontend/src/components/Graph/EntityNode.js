import React, { useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import MetadataList from './MetadataList';
import MetadataValue from './MetadataValue';
import visualizationComponents from '../Visualization/visualizationComponents';

const EntityNode = ({ data }) => {
  // Memoize the visualization component
  const visualizationContent = useMemo(() => {
    if (!data.visualization || !data.visualization.type) return null;
    
    const VisualizationComponent = visualizationComponents[data.visualization.type.toLowerCase()];
    
    if (!VisualizationComponent) {
      console.warn(`No visualization component found for type: ${data.visualization.type}`);
      return null;
    }

    return (
      <div className="w-full p-2">
        <VisualizationComponent visualization={data.visualization} />
      </div>
    );
  }, [data.visualization]); // Only re-render when visualization data changes

  const renderMetadataValue = (value) => {
    if (Array.isArray(value)) {
      return <MetadataList items={value} />;
    }
    return <MetadataValue value={value} />;
  };

  const handleContextMenu = (e) => {
    e.preventDefault();
    navigator.clipboard.writeText(data.path || data.label)
      .then(() => console.log('Path copied to clipboard'))
      .catch(err => console.error('Failed to copy:', err));
  };

  return (
    <div 
      className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 cursor-grab active:cursor-grabbing min-w-[250px]"
      onContextMenu={handleContextMenu}
    >
      <Handle 
        type="target" 
        position={Position.Top} 
        id="top"
        style={{ background: '#4b5563' }}
      />
      <div className="text-white font-medium mb-2">{data.label}</div>
      <div className="space-y-1.5">
        {data.visualization ? (
          visualizationContent
        ) : (
          Object.entries(data.metaData || {}).map(([key, value]) => (
            <div key={key} className="text-sm flex items-start gap-2">
              <span className="text-gray-400">{key}:</span>
              {renderMetadataValue(value)}
            </div>
          ))
        )}
      </div>
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="bottom"
        style={{ background: '#4b5563' }}
      />
    </div>
  );
};

export default React.memo(EntityNode); 