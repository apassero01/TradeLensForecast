import React from 'react';
import EntityNodeBase from './EntityNodeBase';

function DocumentEntity({ data, updateEntity }) {
  const renderContent = ({ 
    data: nodeData, 
    childrenRequests, 
    updateEntity: updateNodeEntity,
    sendStrategyRequest,
    onRemoveRequest,
    isLoading,
    setIsLoading,
    getVisualizationOverlay
  }) => {
    return (
      <div className="relative w-full h-full">
        {/* Base document content - set to lower z-index */}
        <div className="w-full h-full flex items-center justify-center z-0 relative">
          <span className="text-white">{data.content || 'Document'}</span>
        </div>

        {/* Visualization overlay - position absolute and higher z-index */}
        <div className="absolute inset-0 z-20">
          {/* {getVisualizationOverlay()} */}
        </div>
      </div>
    );
  };

  return (
    <EntityNodeBase
      data={data}
      updateEntity={updateEntity}
    >
      {renderContent}
    </EntityNodeBase>
  );
}

export default React.memo(DocumentEntity);
