// BaseEntity.js
import React, { useMemo } from 'react';
import MetadataList from './MetadataList';
import MetadataValue from './MetadataValue';
import visualizationComponents from '../Visualization/visualizationComponents';

function BaseEntity({ entityData }) {
  const handleContextMenu = (e, value) => {
    e.preventDefault();
    e.stopPropagation(); // Stop the event from bubbling up to the entity node
    navigator.clipboard.writeText(String(value)).then(() => {
      // Optional: Add visual feedback
      console.log('Copied to clipboard:', value);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  };

  // If there's a custom visualization, show it; otherwise show metadata
  const content = useMemo(() => {
    const { visualization } = entityData;
    if (visualization?.type) {
      const VisualizationComponent =
        visualizationComponents[visualization.type.toLowerCase()];
      if (!VisualizationComponent) {
        console.warn(`No visualization found for type: ${visualization.type}`);
        return null;
      }
      return (
        <div className="w-full h-full p-4 relative z-10">
          <VisualizationComponent visualization={visualization} />
        </div>
      );
    }

    const metadata = entityData.meta_data || entityData.metaData || {};
    return (
      <div className="p-4 space-y-2">
        {Object.entries(metadata).map(([key, value]) => (
          <div 
            key={key} 
            className="text-sm flex items-start gap-2 rounded px-2 -mx-2"
          >
            <span className="text-gray-400">{key}:</span>
            {Array.isArray(value)
              ? <div className="relative z-20"><MetadataList items={value} /></div>
              : <MetadataValue value={value} />}
          </div>
        ))}
      </div>
    );
  }, [entityData.visualization, entityData.meta_data, entityData.metaData]);

  return (
    <div className="flex flex-col flex-grow min-h-0 overflow-hidden relative">
      {content}
    </div>
  );
}

export default React.memo(BaseEntity, (prev, next) => {
  return (
    prev.entityData.visualization === next.entityData.visualization &&
    prev.entityData.meta_data === next.entityData.meta_data &&
    prev.entityData.metaData === next.entityData.metaData
  );
});