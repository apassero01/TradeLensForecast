// src/hooks/useEntities.js
import { useRecoilState, useRecoilCallback } from 'recoil';
import { entityApi } from '../api/entityApi';
import { entityIdsAtom } from '../state/entityIdsAtom';
import { entityFamily } from '../state/entityFamily';

export const useEntities = () => {
  // We'll manage the array of IDs here
  const [entityIds, setEntityIds] = useRecoilState(entityIdsAtom);

  // 1. Merging a dictionary of { entityId: {...data} } using useRecoilCallback
  //    This ensures we can call "set" on atomFamily inside a top-level hook.
  const mergeEntities = useRecoilCallback(
    ({ set }) =>
      (entityDict) => {
        const newIds = Object.keys(entityDict);

        // Update entityIdsAtom (union with existing)
        setEntityIds((prev) => {
          const idSet = new Set([...prev, ...newIds]);
          return Array.from(idSet);
        });

        // For each ID, partially merge data into the relevant atom
        newIds.forEach((entityId) => {
          set(entityFamily(entityId), (prevData) => ({
            ...prevData,
            ...entityDict[entityId],
          }));
        });
      },
    [setEntityIds]
  );

  // 2. Example of fetching entity data from the API and merging
  async function fetchEntityTypes() {
    try {
      const response = await entityApi.fetchAvailableEntities();
      // Suppose "response" is a dict { entityA: {...}, entityB: {...} }
      mergeEntities(response);
    } catch (error) {
      console.error('Failed to fetch entity types:', error);
      // handle error, e.g., store in a separate error state or session error
    }
  }

  // 3. Clearing entities => just empty the array of IDs
  //    Recoil will garbage-collect any entityFamily atoms no longer referenced
  function clearEntities() {
    setEntityIds([]);
  }

  return {
    entityIds,
    mergeEntities,
    fetchEntityTypes,
    clearEntities,
  };
};