import { selector } from 'recoil';
import { entityIdsAtom } from './entityIdsAtom';
import { entityFamily } from './entityFamily';
import { EntityTypes, NodeTypes } from '../components/Canvas/Entity/EntityEnum';
import { selectorFamily } from 'recoil';

export const strategyRequestChildrenSelector = selectorFamily({
  key: 'strategyRequestChildrenSelector',
  get: (parentId) => ({ get }) => {
    // 1. Get the parent entity
    const parent = get(entityFamily(parentId));
    if (!parent || !parent.child_ids) {
      return [];
    }

    // 2. Retrieve child entities from Recoil
    const children = parent.child_ids.map((childId) => get(entityFamily(childId)));

    // 3. Filter for the ones that are strategy requests
    //    (Adjust this check to match your actual type, e.g. 'STRATEGY_REQUEST')
    return children.filter((child) => child.entity_type === EntityTypes.STRATEGY_REQUEST);
  },
});

export const childrenByTypeSelector = selectorFamily({
  key: 'childrenByTypeSelector',
  get: ({ parentId, type }) => ({ get }) => {
    const parent = get(nodeSelectorFamily(parentId));
    const children = parent.data.child_ids.map((childId) => get(nodeSelectorFamily(childId)));
    const childEntities = children.filter((child) => child.data.entity_type === type);
    return childEntities;
  },
});


export const nodeSelectorFamily = selectorFamily({
  key: 'nodeSelectorFamily',
  get: (entityId) => ({ get }) => {
    const entity = get(entityFamily(entityId));
    if (!entity) return null; // Handle missing entities

    const nodeType = {
      [EntityTypes.STRATEGY_REQUEST]: NodeTypes.STRATEGY_REQUEST_ENTITY,
      [EntityTypes.INPUT]: NodeTypes.INPUT_ENTITY,
      [EntityTypes.VISUALIZATION]: NodeTypes.VISUALIZATION_ENTITY,
      [EntityTypes.DOCUMENT]: NodeTypes.DOCUMENT_ENTITY,
      [EntityTypes.VIEW]: NodeTypes.VIEW_ENTITY,
    }[entity.entity_type] || NodeTypes.ENTITY_NODE;

    return {
      id: entityId,
      type: nodeType,
      position: entity.position || calculateNewPosition(entity, get),
      width: entity?.width || 300,
      height: entity?.height || 200,
      data: { ...entity, entityId },
      hidden: entity?.hidden || false,
      entity_name: entity?.entity_name || '',
      entity_type: entity?.entity_type || '',
      entity_id: entity?.entity_id || '',
      is_loading: false,
    };
  },
});

export const allEntitiesSelector = selector({
  key: 'allEntitiesSelector',
  get: ({ get }) => {
    const ids = get(entityIdsAtom);
    return ids.map((id) => get(nodeSelectorFamily(id)));
  },
});

const calculateNewPosition = (entity, get) => {
  const parent_ids = entity.parent_ids;
  if (parent_ids && parent_ids.length > 0) {
    const parent = get(nodeSelectorFamily(parent_ids[0]));
    const parentPosition = parent.position;
    return {x: parentPosition.x, y: parentPosition.y + 200}
  }
  return {x: 0, y: 0}
}