import { selector } from 'recoil';
import { entityIdsAtom } from './entityIdsAtom';
import { entityFamily } from './entityFamily';

export const flowNodesSelector = selector({
    key: 'flowNodesSelector',
    get: ({ get }) => {
      const entityIds = get(entityIdsAtom);
  
      return entityIds.map(id => {
        const entity = get(entityFamily(id));
        return {
          id,
          // Map entity_type to React Flow node type
          type: entity.entity_type === 'strategy_request' ? 
            'strategyRequestEntity' : 'entityNode',
          position: entity.position || { x: 0, y: 0 },
          data: {
            ...entity,
            entityId: id,
          },
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