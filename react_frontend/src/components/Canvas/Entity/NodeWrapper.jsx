// DynamicNodeWrapper.jsx
import React, { useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../state/entitiesSelectors';
import { EntityTypes, NodeTypes } from './EntityEnum';
import VisualizationEntity from './VisualizationEntity/VisualizationEntity';
import InputEntity from './InputEntity'
import StrategyRequestEntity from './StrategyRequestEntity';
import EntityNode from './EntityNode';
import { entityIdsAtom } from '../../../state/entityIdsAtom';
import { useReactFlow, addEdge } from '@xyflow/react';
import { useRecoilCallback } from 'recoil';
import { entityFamily } from '../../../state/entityFamily';
import ViewEntity from './ViewEntity/ViewEntity';
import DocumentEntity from './DocumentEntity';
import { useWebSocketConsumer } from '../../../hooks/useWebSocketConsumer';
import useUpdateFlowNodes from '../../../hooks/useUpdateFlowNodes';
const componentMapping = {
    [NodeTypes.VISUALIZATION_ENTITY]: VisualizationEntity,
    [NodeTypes.INPUT_ENTITY]: InputEntity,
    [NodeTypes.STRATEGY_REQUEST_ENTITY]: StrategyRequestEntity,
    [NodeTypes.VIEW_ENTITY]: ViewEntity,
    [NodeTypes.DOCUMENT_ENTITY]: DocumentEntity,
};


const DynamicNodeWrapper = ({ id, data, hidden }) => {
    // Always call hooks at the top
    const reactFlowInstance = useReactFlow();
    const nodeData = useRecoilValue(nodeSelectorFamily(id));
    const nodeIds = useRecoilValue(entityIdsAtom);
    const {sendStrategyRequest} = useWebSocketConsumer();
    const updateEntity = useUpdateFlowNodes(reactFlowInstance);

    function updateNodeEdges(nodeId, newParentIds) {
      // 1. Retrieve the current edges from the React Flow instance.
      
      const edges = newParentIds.map((parentId) => ({
        id: `${parentId}-${nodeId}`,
        source: parentId,
        target: nodeId,
      }));
    
      
    
      // 4. Update the edges in the React Flow instance.
      reactFlowInstance.addEdges(edges);
    }


    // Use useEffect unconditionally; if nodeData is not ready, exit early
    useEffect(() => {
        if (!nodeData) return;

        reactFlowInstance.setNodes((nds) =>
            nds.map((node) => {
                if (node.id === id) {
                    // Only update if the positions differ
                    // if (
                    //     node.position.x !== nodeData.position.x ||
                    //     node.position.y !== nodeData.position.y
                    // ) {
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

        updateNodeEdges(id, nodeData.data?.parent_ids);

    }, [nodeData, reactFlowInstance, id, nodeIds, hidden, updateEntity]);

    // Now safely return null if nodeData is not yet available
    if (!nodeData) return null;

    // Compute the dynamic node type based on the node's entity_type.
    let dynamicType;
    switch (nodeData.entity_type) {
        case EntityTypes.STRATEGY_REQUEST:
            dynamicType = NodeTypes.STRATEGY_REQUEST_ENTITY;
            break;
        case EntityTypes.INPUT:
            dynamicType = NodeTypes.INPUT_ENTITY;
            break;
        case EntityTypes.VISUALIZATION:
            dynamicType = NodeTypes.VISUALIZATION_ENTITY;
            break;
        case EntityTypes.VIEW:
            dynamicType = NodeTypes.VIEW_ENTITY;
            break;
        case EntityTypes.DOCUMENT:
            dynamicType = NodeTypes.DOCUMENT_ENTITY;
            break;
        default:
            dynamicType = NodeTypes.ENTITY_NODE;
    }

    // Pick the actual node component based on the computed type.
    const SpecificNodeComponent = componentMapping[dynamicType] || EntityNode;

    // Render the chosen node component with up-to-date properties.
    // Those properties include the backend-provided position, width, height, etc.
    return <SpecificNodeComponent key={`${data.entityId}-${data.hidden}`} id={id} {...nodeData} updateEntity={updateEntity} sendStrategyRequest={sendStrategyRequest} />;
};

export default React.memo(DynamicNodeWrapper);