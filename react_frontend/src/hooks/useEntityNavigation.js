import { useRecoilState } from 'recoil';
import { sessionAtom } from '../state/sessionAtoms';
import { useCallback } from 'react';

/**
 * Hook for navigating between entities at the top-level application layer
 * This updates the session state to change the entire application context
 */
export const useEntityNavigation = () => {
  const [session, setSession] = useRecoilState(sessionAtom);

  const navigateToEntity = useCallback((entityId) => {
    if (!entityId || entityId === session.currentEntityId) return;
    
    setSession(prev => ({
      ...prev,
      viewMode: 'entity',
      currentEntityId: entityId,
    }));
  }, [session.currentEntityId, setSession]);

  const navigateToCanvas = useCallback(() => {
    setSession(prev => ({
      ...prev,
      viewMode: 'canvas',
      currentEntityId: null,
    }));
  }, [setSession]);

  return {
    navigateToEntity,
    navigateToCanvas,
    currentEntityId: session.currentEntityId,
    viewMode: session.viewMode,
    isInEntityView: session.viewMode === 'entity',
  };
};
