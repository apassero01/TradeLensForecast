import { useRecoilValue } from "recoil";
import { childrenByTypeSelector } from "../state/entitiesSelectors";
import { EntityTypes } from "../components/Canvas/Entity/EntityEnum";
import useRenderStoredView from "./useRenderStoredView";


/**
 * 
 * @param {ID of entity with view component} entityId 
 * @param {function to send strategy request} sendStrategyRequest 
 * @param {function to update entity} updateEntity 
 * @param {props to pass to the view} props 
 * @param {type of view component child of entity that we want to return} viewComponentType 
 * @returns {view component child of entity that we want to return} view 
 * This hook is used assuming that we have Entity A that contains a child entity that is a view component of some type. 
 * This hook will return the view component child of entity A that is of the type specified by viewComponentType.
 */
export default function useEntityView(entityId, sendStrategyRequest, updateEntity, props, viewComponentType) {
    const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: entityId, type: EntityTypes.VIEW })) || [];
    const view = viewChildren.find((child) => child.data.view_component_type === viewComponentType);
    return useRenderStoredView(view?.entity_id, sendStrategyRequest, updateEntity, props);
}