import { atom } from 'recoil';

export const sessionAtom = atom({
    key: 'sessionAtom',
    default: {
        isActive: false, 
        sessionId: null,
        sessionName: null, 
        sessionDescription: null,
        savedSessions: [],
        error: null,
        isLoading: false,
        viewMode: 'canvas', // 'canvas' or 'entity'
        currentEntityId: null,
    }
})