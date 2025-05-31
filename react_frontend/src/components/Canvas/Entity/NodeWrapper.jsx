// DynamicNodeWrapper.jsx
import React, { useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../state/entitiesSelectors';
import { EntityTypes } from './EntityEnum';
import VisualizationEntity from './VisualizationEntity/VisualizationEntity';
import InputEntity from './InputEntity'
import StrategyRequestEntity from './StrategyRequestEntity';
import EntityNode from './EntityNode';
import { entityIdsAtom } from '../../../state/entityIdsAtom';
import { useReactFlow, addEdge, useUpdateNodeInternals } from '@xyflow/react';
import ViewEntity from './ViewEntity/ViewEntity';
import DocumentEntity from './DocumentEntity';
import { useWebSocketConsumer } from '../../../hooks/useWebSocketConsumer';
import useUpdateFlowNodes from '../../../hooks/useUpdateFlowNodes';
import RecipeEntity from './RecipeEntity';

const componentMapping = {
    [EntityTypes.VISUALIZATION]: VisualizationEntity,
    [EntityTypes.INPUT]: InputEntity,
    [EntityTypes.STRATEGY_REQUEST]: StrategyRequestEntity,
    [EntityTypes.VIEW]: ViewEntity,
    [EntityTypes.DOCUMENT]: DocumentEntity,
    [EntityTypes.RECIPE]: RecipeEntity,
};


const DynamicNodeWrapper = ({ id, data, hidden }) => {
    // Always call hooks at the top
    const reactFlowInstance = useReactFlow();
    const nodeData = useRecoilValue(nodeSelectorFamily(id));
    const nodeIds = useRecoilValue(entityIdsAtom);
    const {sendStrategyRequest} = useWebSocketConsumer();
    const updateEntity = useUpdateFlowNodes(reactFlowInstance);

    // Use useEffect unconditionally; if nodeData is not ready, exit early
    useEffect(() => {
        if (!nodeData) return;

        reactFlowInstance.setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    return {
                        ...node,
                        position: nodeData.position,
                        width: nodeData.width,
                        height: nodeData.height,
                        hidden: nodeData.hidden,
                    };
                }
                return node;
            })
        );

        // Edge management is now handled centrally in Canvas component
        // Remove the updateNodeEdges call to prevent conflicts

    }, [nodeData, reactFlowInstance, id, nodeIds, hidden, updateEntity]);

    // Now safely return null if nodeData is not yet available
    if (!nodeData) return null;

    // Directly look up the component in the mapping using the entity type.
    // If the type is not found in the mapping, default to EntityNode.
    const SpecificNodeComponent = componentMapping[nodeData.entity_type] || EntityNode;

    // Render the chosen node component with up-to-date properties.
    // Those properties include the backend-provided position, width, height, etc.
    return <SpecificNodeComponent key={`${data.entityId}-${data.hidden}`} id={id} {...nodeData} updateEntity={updateEntity} sendStrategyRequest={sendStrategyRequest} />;
};

export default React.memo(DynamicNodeWrapper);