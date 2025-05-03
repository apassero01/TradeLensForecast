import { selector } from 'recoil';
import { entityIdsAtom } from './entityIdsAtom';
import { entityFamily } from './entityFamily';
import { EntityTypes } from '../components/Canvas/Entity/EntityEnum';
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
    console.log('children', children);
    const childEntities = children.filter((child) => child.data.entity_type === type);
    console.log('childEntities', childEntities);
    return childEntities;
  },
});


export const nodeSelectorFamily = selectorFamily({
  key: 'nodeSelectorFamily',
  get: (entityId) => ({ get }) => {
    const entity = get(entityFamily(entityId));
    if (!entity) return null; // Handle missing entities

    return {
      id: entityId,
      type: entity.entity_type,
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