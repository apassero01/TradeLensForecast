import { useRecoilState, useRecoilValue } from 'recoil';
import { activeAgentIdAtom } from '../state/activeAgentAtom';
import { nodeSelectorFamily } from '../state/entitiesSelectors';

/**
 * Custom hook to manage active agent state across the application
 * @returns {Object} Object containing activeAgentId, activeAgentNode, and setActiveAgentId
 */
export const useActiveAgent = () => {
  const [activeAgentId, setActiveAgentId] = useRecoilState(activeAgentIdAtom);
  const activeAgentNode = useRecoilValue(nodeSelectorFamily(activeAgentId || ''));

  return {
    activeAgentId,
    activeAgentNode: activeAgentNode?.data?.entity_name === 'api_model' ? activeAgentNode : null,
    setActiveAgentId,
    hasActiveAgent: !!activeAgentId && activeAgentNode?.data?.entity_name === 'api_model',
  };
};
