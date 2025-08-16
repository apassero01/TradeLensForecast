import React, { useCallback, useEffect, useState, useRef, useMemo } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { WindowFrame } from './WindowFrame';
import { EntityExplorer } from './EntityExplorer';
import { useWindowManager } from './WindowManager';
import { AgentDashboard } from './AgentDashboard';

interface ComposableDesktopViewProps {
  data?: any;
  sendStrategyRequest: (request: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

export const ComposableDesktopView: React.FC<ComposableDesktopViewProps> = ({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}) => {
  // Get view entity data for persistence
  const viewEntity: any = useRecoilValue(nodeSelectorFamily(viewEntityId));
  
  // Debounce timer refs
  const persistWindowsTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isUserActionInProgressRef = useRef<boolean>(false);
  const hasInitializedRef = useRef<boolean>(false);

  // Initialize the window manager without any initial windows - clean desktop
  const {
    windows,
    openWindow,
    closeWindow,
    focusWindow,
    updateWindowPosition,
    updateWindowSize,
    loadPersistedWindows,
  } = useWindowManager();

  // Get pinned entities directly from backend (no local state)
  const pinnedEntities = useMemo(() => 
    viewEntity?.data?.pinned_entities || viewEntity?.data?.attribute_map?.pinned_entities || [],
    [viewEntity?.data?.pinned_entities, viewEntity?.data?.attribute_map?.pinned_entities]
  );

  // Load persisted windows only on initial mount
  useEffect(() => {
    // Only initialize once when component mounts
    if (hasInitializedRef.current) {
      return;
    }
    
    if (viewEntity?.data) {
      const persistedWindows = viewEntity.data.desktop_windows || viewEntity.data.attribute_map?.desktop_windows || [];
      
      if (Array.isArray(persistedWindows) && persistedWindows.length > 0) {
        loadPersistedWindows(persistedWindows);
      }
      
      hasInitializedRef.current = true;
    }
  }, [viewEntity?.data, loadPersistedWindows]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (persistWindowsTimeoutRef.current) {
        clearTimeout(persistWindowsTimeoutRef.current);
      }
    };
  }, []);

  // Debounced helper function to persist windows state
  const persistWindowsStateDebounced = useCallback((delay: number = 100) => {
    // Mark that user action is in progress
    isUserActionInProgressRef.current = true;
    
    // Clear existing timeout
    if (persistWindowsTimeoutRef.current) {
      clearTimeout(persistWindowsTimeoutRef.current);
    }
    
    // Set new timeout
    persistWindowsTimeoutRef.current = setTimeout(() => {
      const windowsToSave = windows.map(w => ({
        entityId: w.entityId,
        position: w.position,
        size: w.size,
        windowType: w.entityId.startsWith('explorer-') ? 'explorer' : 
                   w.entityId === 'agent-dashboard' ? 'agent-dashboard' : 'entity'
      }));
      
      const request = StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(viewEntityId)
        .withParams({ 
          attribute_map: { 
            desktop_windows: windowsToSave 
          } 
        })
        .withAddToHistory(false)
        .build();
      
      sendStrategyRequest(request);
      persistWindowsTimeoutRef.current = null;
      
      // Clear user action flag after persistence and a small delay
      setTimeout(() => {
        isUserActionInProgressRef.current = false;
      }, 100);
    }, delay);
  }, [windows, viewEntityId, sendStrategyRequest]);

  const handleEntityDoubleClick = useCallback((entityId: string) => {
    openWindow(entityId);
    // Persist after opening window
    persistWindowsStateDebounced(100);
  }, [openWindow, persistWindowsStateDebounced]);

  const handlePinEntity = useCallback((entityId: string) => {
    const currentPinned = Array.isArray(pinnedEntities) ? pinnedEntities : [];
    if (!currentPinned.includes(entityId)) {
      const newPinned = [...currentPinned, entityId];
      
      const request = StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(viewEntityId)
        .withParams({ 
          attribute_map: { 
            pinned_entities: newPinned 
          } 
        })
        .withAddToHistory(false)
        .build();
      
      sendStrategyRequest(request);
    }
  }, [pinnedEntities, viewEntityId, sendStrategyRequest]);

  const handleUnpinEntity = useCallback((entityId: string) => {
    const currentPinned = Array.isArray(pinnedEntities) ? pinnedEntities : [];
    const newPinned = currentPinned.filter(id => id !== entityId);
    
    const request = StrategyRequests.builder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(viewEntityId)
      .withParams({ 
        attribute_map: { 
          pinned_entities: newPinned 
        } 
      })
      .withAddToHistory(false)
      .build();
    
    sendStrategyRequest(request);
  }, [pinnedEntities, viewEntityId, sendStrategyRequest]);

  const handleWindowClose = useCallback((windowId: string) => {
    closeWindow(windowId);
    // Persist after closing window
    persistWindowsStateDebounced(100);
  }, [closeWindow, persistWindowsStateDebounced]);

  const handleWindowFocus = useCallback((windowId: string) => {
    focusWindow(windowId);
  }, [focusWindow]);

  const handleWindowPositionChange = useCallback((windowId: string, position: { x: number; y: number }) => {
    updateWindowPosition(windowId, position);
    // Persist after position change (debounced)
    persistWindowsStateDebounced(1000);
  }, [updateWindowPosition, persistWindowsStateDebounced]);

  const handleWindowSizeChange = useCallback((windowId: string, size: { width: number; height: number }) => {
    updateWindowSize(windowId, size);
    // Persist after size change (debounced)
    persistWindowsStateDebounced(1000);
  }, [updateWindowSize, persistWindowsStateDebounced]);

  // Pinned Entity Icon Component
  const PinnedEntityIcon: React.FC<{ 
    entityId: string; 
    onClick: () => void; 
    onUnpin: () => void;
  }> = ({ entityId, onClick, onUnpin }) => {
    const entityNode: any = useRecoilValue(nodeSelectorFamily(entityId));
    const [showUnpinButton, setShowUnpinButton] = useState(false);

    if (!entityNode?.data) {
      return null;
    }

    const entityType = entityNode.data.entity_type || 'entity';
    const displayName = entityNode.data.name || 
                       entityNode.data.entity_name || 
                       entityNode.data.display_name ||
                       entityNode.data.title ||
                       entityType;

    // Get entity icon
    const getEntityIcon = (type: string): string => {
      const iconMap: Record<string, string> = {
        'api_model': '🤖',
        'agent': '🤖',
        'document': '📄',
        'file': '📁',
        'folder': '📁',
        'view': '👁️',
        'chat': '💬',
        'message': '💬',
        'user': '👤',
        'data': '💾',
        'model': '🧠',
        'training': '🎯',
        'session': '🔗',
        'entity': '⚡',
      };
      return iconMap[type.toLowerCase()] || '📦';
    };

    return (
      <div 
        className="relative"
        onMouseEnter={() => setShowUnpinButton(true)}
        onMouseLeave={() => setShowUnpinButton(false)}
      >
        <button
          onClick={onClick}
          className="w-12 h-12 bg-gray-700 hover:bg-gray-600 rounded-xl flex items-center justify-center text-xl transition-all transform hover:scale-110 active:scale-95"
          title={displayName}
        >
          {getEntityIcon(entityType)}
        </button>
        
        {/* Unpin button */}
        {showUnpinButton && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onUnpin();
            }}
            className="absolute -top-1 -right-1 w-5 h-5 bg-red-600 hover:bg-red-500 rounded-full text-white text-xs flex items-center justify-center transition-all"
            title="Unpin from dock"
          >
            ×
          </button>
        )}
      </div>
    );
  };

  // Dock/Taskbar Component
  const Dock: React.FC = () => {
    const handleOpenEntityExplorer = () => {
      // Open Entity Explorer as a window
      openWindow(`explorer-${parentEntityId}`);
      // Persist after opening explorer
      persistWindowsStateDebounced(100);
    };

    const handleOpenAgentDashboard = () => {
      // Open Agent Dashboard as a window
      openWindow('agent-dashboard');
      // Persist after opening agent dashboard
      persistWindowsStateDebounced(100);
    };

    const handlePinnedEntityClick = (entityId: string) => {
      openWindow(entityId);
      // Persist after opening pinned entity
      persistWindowsStateDebounced(100);
    };

    return (
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-30">
        <div className="bg-gray-800/90 backdrop-blur-md border border-gray-600/50 rounded-2xl px-4 py-3 shadow-2xl">
          <div className="flex items-center gap-3">
            {/* Entity Explorer Icon */}
            <button
              onClick={handleOpenEntityExplorer}
              className="w-12 h-12 bg-blue-600 hover:bg-blue-500 rounded-xl flex items-center justify-center text-white text-xl transition-all transform hover:scale-110 active:scale-95"
              title="Entity Explorer"
            >
              🌳
            </button>
            
            {/* Agent Dashboard Icon */}
            <button
              onClick={handleOpenAgentDashboard}
              className="w-12 h-12 bg-purple-600 hover:bg-purple-500 rounded-xl flex items-center justify-center text-white text-xl transition-all transform hover:scale-110 active:scale-95"
              title="Agent Dashboard"
            >
              🤖
            </button>
            
            {/* Separator */}
            {Array.isArray(pinnedEntities) && pinnedEntities.length > 0 && (
              <div className="w-px h-8 bg-gray-600/50" />
            )}
            
            {/* Pinned Entities */}
            {Array.isArray(pinnedEntities) && pinnedEntities.map((entityId) => (
              <PinnedEntityIcon
                key={entityId}
                entityId={entityId}
                onClick={() => handlePinnedEntityClick(entityId)}
                onUnpin={() => handleUnpinEntity(entityId)}
              />
            ))}
            
            {/* Window indicators for open windows */}
            {windows.length > 0 && (
              <>
                <div className="w-px h-8 bg-gray-600/50" />
                <div className="flex gap-1">
                  {windows.map((window) => (
                    <button
                      key={window.id}
                      onClick={() => focusWindow(window.id)}
                      className="w-3 h-3 bg-white/60 hover:bg-white/80 rounded-full transition-all"
                      title={`Window: ${window.entityId.slice(0, 8)}...`}
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="relative w-full h-full bg-gradient-to-br from-gray-800 via-gray-900 to-black overflow-hidden">
      {/* Desktop Background with Subtle Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div 
          className="w-full h-full"
          style={{
            backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.15) 1px, transparent 0)',
            backgroundSize: '20px 20px',
          }}
        />
      </div>

      {/* Desktop Info Bar */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-900/80 backdrop-blur-sm border-b border-gray-700 px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-lg">🖥️</div>
            <h1 className="text-sm font-semibold text-white">Composable Desktop</h1>
            <div className="text-xs text-gray-400">
              Entity-centric workspace
            </div>
          </div>
          
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <div>
              {windows.length} window{windows.length !== 1 ? 's' : ''} open
            </div>
            <div className="text-gray-600">
              Root: {parentEntityId.slice(0, 8)}...
            </div>
          </div>
        </div>
      </div>

      {/* Window Container */}
      <div className="absolute inset-0 pt-12 pb-20">
        {windows.map((window) => {
          // Check if this is the EntityExplorer window (starts with 'explorer-')
          const isExplorerWindow = window.entityId.startsWith('explorer-');
          // Check if this is the Agent Dashboard window
          const isAgentDashboardWindow = window.entityId === 'agent-dashboard';
          
          if (isExplorerWindow) {
            // Extract the actual parent entity ID from the explorer window ID
            const actualEntityId = window.entityId.replace('explorer-', '');
            
            // Render EntityExplorer as a draggable window
            return (
              <WindowFrame
                key={window.id}
                windowId={window.id}
                entityId={actualEntityId}
                position={window.position}
                size={window.size}
                zIndex={window.zIndex}
                onClose={handleWindowClose}
                onFocus={handleWindowFocus}
                onPositionChange={handleWindowPositionChange}
                onSizeChange={handleWindowSizeChange}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity}
                customContent={
                  <EntityExplorer
                    rootEntityId={actualEntityId}
                    onEntityDoubleClick={handleEntityDoubleClick}
                    onEntityPin={handlePinEntity}
                    sendStrategyRequest={sendStrategyRequest}
                    viewEntityId={viewEntityId}
                  />
                }
                customTitle="Entity Explorer"
                customIcon="🌳"
              />
            );
          }

          if (isAgentDashboardWindow) {
            // Render Agent Dashboard as a draggable window
            return (
              <WindowFrame
                key={window.id}
                windowId={window.id}
                entityId={window.entityId}
                position={window.position}
                size={window.size}
                zIndex={window.zIndex}
                onClose={handleWindowClose}
                onFocus={handleWindowFocus}
                onPositionChange={handleWindowPositionChange}
                onSizeChange={handleWindowSizeChange}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity}
                customContent={
                  <AgentDashboard
                    sendStrategyRequest={sendStrategyRequest}
                    onEntityDoubleClick={handleEntityDoubleClick}
                  />
                }
                customTitle="Agent Dashboard"
                customIcon="🤖"
              />
            );
          }

          // Render regular entity windows using WindowFrame
          return (
            <WindowFrame
              key={window.id}
              windowId={window.id}
              entityId={window.entityId}
              position={window.position}
              size={window.size}
              zIndex={window.zIndex}
              onClose={handleWindowClose}
              onFocus={handleWindowFocus}
              onPositionChange={handleWindowPositionChange}
              onSizeChange={handleWindowSizeChange}
              sendStrategyRequest={sendStrategyRequest}
              updateEntity={updateEntity}
            />
          );
        })}
      </div>

      {/* Empty State */}
      {windows.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pb-20">
          <div className="text-center text-gray-500">
            <div className="text-6xl mb-4">🖥️</div>
            <h2 className="text-xl font-semibold mb-2 text-gray-400">Welcome to Composable Desktop</h2>
            <p className="text-sm mb-4 max-w-md">
              This is an entity-centric desktop environment. Click the 🌳 icon in the dock below to start exploring your entities.
            </p>
            <div className="text-xs text-gray-600">
              Drag windows around • Resize from corners • Double-click entities to open them
            </div>
          </div>
        </div>
      )}

      {/* macOS-style Dock */}
      <Dock />
    </div>
  );
};

export default ComposableDesktopView;
