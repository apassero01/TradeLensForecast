import { atom, selector } from 'recoil';
import { entityApi } from '../api/entityApi';

// 1. Create a trigger atom to force a refresh
export const refreshTriggerAtom = atom({
  key: 'refreshTriggerAtom',
  default: 0,
});

// 2. Create an async selector that fetches data from the API.
// It depends on the refreshTriggerAtom so that when it changes, the API call re-runs.
export const registrySelector = selector({
  key: 'registrySelector',
  get: async ({ get }) => {
    // Depend on refreshTriggerAtom. Its value is not used except to trigger a re-run.
    get(refreshTriggerAtom);
    const result = await entityApi.getStrategyRegistry();
    return Object.values(result).flat();
  },
});

// 3. Create an async selector for available entities
export const availableEntitiesSelector = selector({
  key: 'availableEntitiesSelector',
  get: async ({ get }) => {
    // Depend on refreshTriggerAtom. Its value is not used except to trigger a re-run.
    get(refreshTriggerAtom);
    try {
      const result = await entityApi.fetchAvailableEntities();
      return result;
    } catch (error) {
      console.error('Failed to fetch available entities:', error);
      return {};
    }
  },
});

