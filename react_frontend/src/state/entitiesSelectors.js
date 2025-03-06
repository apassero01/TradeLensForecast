import { selector } from 'recoil';
import { entityIdsAtom } from './entityIdsAtom';
import { entityFamily } from './entityFamily';
import { EntityTypes, NodeTypes } from '../components/Canvas/Entity/EntityEnum';
import { selectorFamily } from 'recoil';

const entityCache = new Map();

export const flowNodesSelector = selector({
  key: 'flowNodesSelector',
  get: ({ get }) => {
    const entityIds = get(entityIdsAtom);
    // Using a simple cache

    return entityIds.map(id => {
      const entity = get(entityFamily(id));
      const cached = entityCache.get(id);
      // Shallow compare stable properties (ignoring transient ones)
      if (cached) {
        const { position, width, height, ...stableEntity } = entity;
        const { position: cachedPosition, width: cachedWidth, height: cachedHeight, ...cachedStable } = cached;
        if (JSON.stringify(cachedStable) === JSON.stringify(stableEntity)) {
          return cached;
        }
      }

      console.log('COMPUTING NODE SELECTOR FOR ENTITY', entity.entity_type)
      let nodeType;
      switch (entity.entity_type) {
        case EntityTypes.STRATEGY_REQUEST:
          nodeType = NodeTypes.STRATEGY_REQUEST_ENTITY;
          break;
        case EntityTypes.INPUT:
          nodeType = NodeTypes.INPUT_ENTITY;
          break;
        case EntityTypes.VISUALIZATION:
          nodeType = NodeTypes.VISUALIZATION_ENTITY;
          break;
        default:
          nodeType = NodeTypes.ENTITY_NODE;
      }
      const nodeData = {
        id,
        type: nodeType,
        position: entity.position || { x: 0, y: 0 },
        data: {
          ...entity,
          entityId: id,
        },
        hidden: entity.hidden || false,
      };
      entityCache.set(id, nodeData);
      return nodeData;
    });
  },
});

  export const flowEdgesSelector = selector({
    key: 'flowEdgesSelector',
    get: ({ get }) => {
      const ids = get(entityIdsAtom);
      const edges = [];
      const edgeSet = new Set();
  
      ids.forEach((entityId) => {
        const { child_ids = [] } = get(entityFamily(entityId));
        child_ids.forEach((childId) => {
          const edgeKey = `${entityId}->${childId}`;
          if (!edgeSet.has(edgeKey)) {
            edgeSet.add(edgeKey);
            edges.push({
              id: edgeKey,
              source: entityId,
              target: childId,
              type: 'default',
            });
          }
        });
      });
      return edges;
    },
  });

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


  export const nonTransientEntitySelector = selectorFamily({
    key: 'nonTransientEntitySelector',
    get: (entityId) => ({ get }) => {
      const entity = get(entityFamily(entityId));
      // Destructure to remove transient properties
      const { position, width, height, ...stableData } = entity;
      return stableData;
    },
  });