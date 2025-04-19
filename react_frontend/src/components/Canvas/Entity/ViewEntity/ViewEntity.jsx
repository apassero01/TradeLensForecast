import React, { memo, useMemo, useEffect } from 'react';
import EntityNodeBase from '../EntityNodeBase';
import visualizationComponents from '../VisualizationEntity/Visualization/visualizationComponents';
import ErrorBoundary from '../../../common/ErrorBoundary';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../state/entitiesSelectors';
import useRenderStoredView from '../../../../hooks/useRenderStoredView';
// No longer need useSetRecoilState or entityFamily here

function ViewEntity({ data, sendStrategyRequest, updateEntity }) {

    // --- Hooks at Top Level ---
    const parentIds = data.parent_ids;
    const parentEntityId = parentIds?.length > 0 ? parentIds[0] : null;
    const parentEntity = useRecoilValue(nodeSelectorFamily(parentEntityId));

    // const viewData = useMemo(() => {
    //     if (!parentEntity || !parentEntity.data || !data.parent_attributes) {
    //         return {};
    //     }
    //     return Object.entries(data.parent_attributes).reduce((acc, [parentAttrKey, newKey]) => {
    //         if (parentEntity.data.hasOwnProperty(parentAttrKey)) {
    //             acc[newKey] = parentEntity.data[parentAttrKey];
    //         }
    //         return acc;
    //     }, {});
    // }, [parentEntity, data.parent_attributes]);

    // const visualizationType = data.view_component_type;
    // const VisualizationComponent = visualizationType
    //     ? visualizationComponents[visualizationType]
    //     : null;

    // const completeVisualizationProps = useMemo(() => ({
    //     visualization: {data: viewData},
    //     entityId: data.entityId,
    //     parent_ids: data.parent_ids,
    //     // viewData: viewData,
    // }), [
    //     viewData,
    //     data.entityId,
    //     data.parent_ids,
    //     // viewData,
    // ]);

    // useEffect(() => {
    //     let newDetails = null;
    //     if (visualizationType && VisualizationComponent) {
    //         newDetails = {
    //             type: visualizationType,
    //             props: completeVisualizationProps
    //         };
    //     }
    //     updateEntity(data.entityId, { currentView: newDetails });
    //     return () => {
    //         updateEntity(data.entityId, { currentView: null });
    //     };
    // }, [
    //     data.entityId,
    //     visualizationType,
    //     VisualizationComponent,
    //     completeVisualizationProps,
    //     updateEntity
    // ]);

    const viewData = useRenderStoredView(data.entityId, sendStrategyRequest, updateEntity);


    // --- Render with EntityNodeBase using Render Prop ---
    return (
        <EntityNodeBase
            data={data}
            updateEntity={updateEntity} // Pass if needed by EntityNodeBase itself
        >
            {/* Provide a function as children */}
            {({ /* You can destructure props from EntityNodeBase here if needed later */ }) => (
                 // Move the JSX rendering logic inside the render prop function
                <div className="flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
                    <div className="h-full w-full px-6 overflow-hidden">
                        {viewData}
                    </div>
                </div>
            )}
        </EntityNodeBase>
    );
}

export default memo(ViewEntity);