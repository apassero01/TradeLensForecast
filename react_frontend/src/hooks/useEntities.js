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
        
        // Find IDs that have deleted: true
        const deletedIds = newIds.filter(id => entityDict[id].deleted === true);
        
        // Find IDs that are not marked as deleted
        const validNewIds = newIds.filter(id => entityDict[id].deleted !== true);
        
        // Update entityIdsAtom to remove deleted IDs and add new valid IDs
        setEntityIds((prev) => {
          // Remove any IDs that are marked for deletion
          const remainingIds = prev.filter(id => !deletedIds.includes(id));
          
          // Find IDs that are not already in the remaining array
          const idsToAdd = validNewIds.filter((id) => !remainingIds.includes(id));
          
          if (deletedIds.length === 0 && idsToAdd.length === 0) {
            // No changes needed, return the same array
            return prev;
          }
          
          // Return a new array with deleted IDs removed and new IDs added
          return [...remainingIds, ...idsToAdd];
        });
        
        // For each ID, merge the entity data into the relevant atom
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