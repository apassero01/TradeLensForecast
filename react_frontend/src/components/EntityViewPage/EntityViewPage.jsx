import React from 'react';
import { useParams } from 'react-router-dom';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../state/entitiesSelectors';
import EntityViewRenderer from '../Canvas/Entity/ViewEntity/View/ChatInterface/EntityViewRenderer';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
import { useSession } from '../../hooks/useSession';

/**
 * Full-screen dedicated view for individual entities
 * Accessible via /entity/<entity_id> route
 */
const EntityViewPage = () => {
  const { entityId } = useParams();
  const entity = useRecoilValue(nodeSelectorFamily(entityId));
  const { sendStrategyRequest, ws } = useWebSocketConsumer();
  const { isActive } = useSession();
  
  // Debug logging
  console.log('EntityViewPage - entityId:', entityId);
  console.log('EntityViewPage - entity from nodeSelectorFamily:', entity);
  console.log('EntityViewPage - entity.data:', entity?.data);
  console.log('EntityViewPage - entity.data.child_ids:', entity?.data?.child_ids);
  console.log('EntityViewPage - WebSocket connection:', ws);
  console.log('EntityViewPage - WebSocket readyState:', ws?.readyState);
  console.log('EntityViewPage - Session isActive:', isActive);

  // Handle entity updates
  const updateEntity = (entityId, data) => {
    console.log('Entity update requested:', entityId, data);
    // Could implement entity updates here if needed
  };

  // Loading state
  if (!entity || !entity.data) {
    return (
      <div className="flex-1 bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-gray-400">Loading entity...</p>
          <p className="text-xs text-gray-500 font-mono">ID: {entityId}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gray-900 text-white overflow-auto">
      {/* Main content area - full screen entity display */}
      <div className="h-full">
        <EntityViewRenderer
          entityId={entityId}
          sendStrategyRequest={sendStrategyRequest}
          updateEntity={updateEntity}
        />
      </div>
    </div>
  );
};

export default EntityViewPage;