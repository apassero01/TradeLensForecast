import { selector } from 'recoil';
import { entityIdsAtom } from './entityIdsAtom';
import { entityFamily } from './entityFamily';
import { EntityTypes, NodeTypes } from '../components/Canvas/Entity/EntityEnum';
import { selectorFamily } from 'recoil';
export const flowNodesSelector = selector({
  key: 'flowNodesSelector',
  get: ({ get }) => {
    const entityIds = get(entityIdsAtom);

    return entityIds.map(id => {
      const entity = get(entityFamily(id));
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

      return {
        id,
        type: nodeType,
        position: entity.position || { x: 0, y: 0 },
        data: {
          ...entity,
          entityId: id,
        },
        hidden: entity.hidden || false,
      };
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