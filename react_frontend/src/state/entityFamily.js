import { atomFamily } from 'recoil';

// Each entity has the same shape: entity_name, position, etc.
// The parameter to atomFamily is the entityId (string).
export const entityFamily = atomFamily({
  key: 'entityFamily',
  default: (entityId) => ({
    entity_id: entityId,
    entity_name: '',
    entity_type: '',
    position: { x: 0, y: 0 },
    width: 200,
    height: 100,
    child_ids: [],
    strategy_requests: [],
    // etc. for other fields from the back end
  }),
});