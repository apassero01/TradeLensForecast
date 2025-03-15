import React, { memo, useCallback } from 'react';
import EntityNodeBase from '../EntityNodeBase';
import visualizationComponents from './Visualization/visualizationComponents';
import { useRecoilValue } from 'recoil';

function VisualizationEntity({ data, sendStrategyRequest }) {
    

    const VisualizationComponent = data.visualization?.type 
        ? visualizationComponents[data.visualization.type]
        : null;

    return (
        <EntityNodeBase 
            data={data}
        >
            {({ sendStrategyRequest }) => (
                <div className="flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
                    <div className="h-full w-full px-6 overflow-hidden">
                        {VisualizationComponent ? (
                            <VisualizationComponent 
                                visualization={data.visualization} 
                                sendStrategyRequest={sendStrategyRequest}
                                entityId={data.entityId}
                                parent_ids={data.parent_ids}
                            />
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