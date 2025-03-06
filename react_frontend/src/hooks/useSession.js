import { useRecoilState, useResetRecoilState } from 'recoil';
import { sessionAtom } from '../state/sessionAtoms';
import { useEntities } from './useEntities';
import { entityApi } from '../api/entityApi';
import { useWebSocket } from './useWebSocket';
export const useSession = () => {
    const [session, setSession] = useRecoilState(sessionAtom);
    const resetSession = useResetRecoilState(sessionAtom);
    

    const { entityIds, mergeEntities, fetchEntityTypes, clearEntities } = useEntities();
    const { isActive, isLoading, savedSessions, sessionId, error } = session;

    function setSessionError(newError) {
        setSession(prev => ({ ...prev, error: newError }));
      }    

    async function startSession() {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            const data = await entityApi.startSession();
            const sessionData = data.sessionData;
            // This is ugly and will be fixed after we migrate to new frontend  
            const sessionId = Object.keys(sessionData)[0];
            console.log(sessionData)
            resetSession();
            mergeEntities(sessionData);
            console.log(sessionData)

        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    async function stopSession() {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            await entityApi.stopSession();
            resetSession();
            clearEntities();
            fetchSavedSessions();
        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    async function saveSession() {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            await entityApi.saveSession();
            setSession(prev => ({ ...prev, isLoading: false, error: null}));
        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    async function loadSession(sessionId) {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            resetSession();
            clearEntities();
            const data = await entityApi.loadSession(sessionId);
            mergeEntities(data.session_data);
            setSession(prev => ({ ...prev, isLoading: false, error: null, sessionId: sessionId, isActive: true}));
        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    async function deleteSession() {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            await entityApi.deleteSession();
            resetSession();
            clearEntities();
        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    async function fetchSavedSessions() {
        setSession(prev => ({ ...prev, isLoading: true, error: null}));
        try {
            const data = await entityApi.getSavedSessions();
            setSession(prev => ({ ...prev, isLoading: false, error: null, savedSessions: data.sessions}));
        } catch (error) {
            setSession(prev => ({ ...prev, isLoading: false, error: error.message}));
        }
    }

    return {
        sessionId,
        isActive,
        isLoading,
        savedSessions, 
        startSession,
        stopSession,
        saveSession,
        loadSession,
        deleteSession,
        fetchSavedSessions,
        error,
        setSessionError,
    }
}