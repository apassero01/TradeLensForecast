import React, { memo } from 'react';
import EntityNodeBase from '../EntityNodeBase';
import visualizationComponents from './Visualization/visualizationComponents';
import ErrorBoundary from '../../../common/ErrorBoundary';

function VisualizationEntity({ data, sendStrategyRequest, updateEntity }) {
    

    const VisualizationComponent = data.visualization?.type 
        ? visualizationComponents[data.visualization.type]
        : null;

    return (
        <EntityNodeBase 
            data={data}
            updateEntity={updateEntity}
        >
            {({ sendStrategyRequest, updateEntity }) => (
                <div className="flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
                    <div className="h-full w-full px-6 overflow-hidden">
                        {VisualizationComponent ? (
                            <ErrorBoundary 
                                fallback={error => (
                                    <div className="text-red-500 text-sm p-4 border border-red-300 rounded bg-red-50">
                                        <div className="font-bold mb-1">Error loading visualization:</div>
                                        <div>{error?.message || "Unknown error"}</div>
                                    </div>
                                )}
                            >
                                <VisualizationComponent 
                                    visualization={data.visualization} 
                                    sendStrategyRequest={sendStrategyRequest}
                                    updateEntity={updateEntity}
                                    entityId={data.entityId}
                                    parent_ids={data.parent_ids}
                                />
                            </ErrorBoundary>
                        ) : (
                            <div className="text-gray-400 text-sm">No visualization available</div>
                        )}
                    </div>
                </div>
            )}
        </EntityNodeBase>
    );
}

export default memo(VisualizationEntity);