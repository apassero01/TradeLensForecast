import React, { useMemo } from 'react';
import { Handle, Position } from 'reactflow';
import MetadataList from './MetadataList';
import MetadataValue from './MetadataValue';
import visualizationComponents from '../Visualization/visualizationComponents';

const EntityNode = React.memo(({ data }) => {
  // Memoize the entire content to prevent re-renders during drag
  const content = useMemo(() => {
    console.log('EntityNode content being recalculated for:', data.entity_name + data.id);
    
    // Memoize the visualization component
    const visualizationContent = data.visualization?.type ? (
      (() => {
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
      })()
    ) : null;

    // Get metadata from either meta_data or metaData
    const metadata = data.meta_data || data.metaData || {};

    const renderMetadataValue = (value) => {
      if (Array.isArray(value)) {
        return <MetadataList items={value} />;
      }
      return <MetadataValue value={value} />;
    };

    return (
      <>
        <div className="text-white font-medium mb-2">{data.entity_name}</div>
        <div className="space-y-1.5">
          {data.visualization ? (
            visualizationContent
          ) : (
            Object.entries(metadata).map(([key, value]) => (
              <div key={key} className="text-sm flex items-start gap-2">
                <span className="text-gray-400">{key}:</span>
                {renderMetadataValue(value)}
              </div>
            ))
          )}
        </div>
      </>
    );
  }, [data.entity_name, data.visualization, data.meta_data, data.metaData]); // Only re-render when these change

  const handleContextMenu = (e) => {
    e.preventDefault();
    // Copy entity ID instead of path
    navigator.clipboard.writeText(data.id)
      .then(() => console.log('Entity ID copied to clipboard:', data.id))
      .catch(err => console.error('Failed to copy ID:', err));
  };

  return (
    <div 
      className="px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 cursor-grab active:cursor-grabbing min-w-[250px]"
      onContextMenu={handleContextMenu}
      title={`Right click to copy ID: ${data.id}`} // Added tooltip
    >
      <Handle 
        type="target" 
        position={Position.Top} 
        id="top"
        style={{ background: '#4b5563' }}
      />
      {content}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        id="bottom"
        style={{ background: '#4b5563' }}
      />
    </div>
  );
}, (prev, next) => {
  // Custom comparison function for React.memo
  // Only re-render if these specific properties change
  return (
    prev.data.entity_name === next.data.entity_name &&
    prev.data.meta_data === next.data.meta_data &&
    prev.data.visualization === next.data.visualization
  );
});

export default EntityNode; 