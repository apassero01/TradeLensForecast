import React from 'react';
import { useRecoilValue } from 'recoil';
import { sessionAtom } from '../state/sessionAtoms';
import TopBar from './TopBar/TopBar';
import Canvas from './Canvas/Canvas';
import EntityViewRenderer from './Canvas/Entity/ViewEntity/View/ChatInterface/EntityViewRenderer';
import { useWebSocketConsumer } from '../hooks/useWebSocketConsumer';
import { FaArrowLeft } from 'react-icons/fa';
import { useRecoilState } from 'recoil';

function ViewSwitcher() {
  const [session, setSession] = useRecoilState(sessionAtom);
  const { sendStrategyRequest } = useWebSocketConsumer();
  

  const updateEntity = (entityId, data) => {
    console.log('Entity update requested:', entityId, data);
  };

  if (session.viewMode === 'entity' && session.currentEntityId) {
    return (
      <div className="flex-1 bg-gray-900 text-white overflow-hidden">
        <EntityViewRenderer
          entityId={session.currentEntityId}
          sendStrategyRequest={sendStrategyRequest}
          updateEntity={updateEntity}
        />
      </div>
    );
  }

  // Default canvas view
  return (
    <>
      <div className="flex-shrink-0">
        <TopBar />
      </div>
      <div className="flex-1 min-h-0">
        <Canvas />
      </div>
    </>
  );
}

export default ViewSwitcher;