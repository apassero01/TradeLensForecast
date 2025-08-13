import { atom, selector } from 'recoil';

// Atom to store the currently active agent ID
export const activeAgentIdAtom = atom({
  key: 'activeAgentId',
  default: null, // No active agent by default
});

// Selector to get active agent data (if needed)
export const activeAgentSelector = selector({
  key: 'activeAgent',
  get: ({ get }) => {
    const activeId = get(activeAgentIdAtom);
    // Could potentially fetch agent data here if needed
    return activeId;
  },
});
