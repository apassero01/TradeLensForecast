import React, { useCallback } from 'react';
import { Rnd } from 'react-rnd';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../../../state/entitiesSelectors';
import EntityViewRenderer from '../ChatInterface/EntityViewRenderer';
import { Position, Size } from './WindowManager';

interface WindowFrameProps {
  windowId: string;
  entityId: string;
  position: Position;
  size: Size;
  zIndex: number;
  onClose: (windowId: string) => void;
  onFocus: (windowId: string) => void;
  onPositionChange: (windowId: string, position: Position) => void;
  onSizeChange: (windowId: string, size: Size) => void;
  sendStrategyRequest: (request: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  customContent?: React.ReactNode;
  customTitle?: string;
  customIcon?: string;
}

// Helper function to get entity icon
function getEntityIcon(type: string): string {
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
    'recipe': '🍳',
    'meal': '🍽️',
    'calendar': '📅',
    'event': '📅',
    'task': '✅',
    'note': '📝',
    'image': '🖼️',
    'video': '🎥',
    'audio': '🎵',
  };
  
  return iconMap[type.toLowerCase()] || '📦';
}

export const WindowFrame: React.FC<WindowFrameProps> = ({
  windowId,
  entityId,
  position,
  size,
  zIndex,
  onClose,
  onFocus,
  onPositionChange,
  onSizeChange,
  sendStrategyRequest,
  updateEntity,
  customContent,
  customTitle,
  customIcon,
}) => {
  const node: any = useRecoilValue(nodeSelectorFamily(entityId));

  const handleDragStop = useCallback((e: any, data: any) => {
    onPositionChange(windowId, { x: data.x, y: data.y });
  }, [windowId, onPositionChange]);

  const handleResizeStop = useCallback((e: any, direction: any, ref: any, delta: any, position: any) => {
    onSizeChange(windowId, {
      width: ref.offsetWidth,
      height: ref.offsetHeight,
    });
    onPositionChange(windowId, position);
  }, [windowId, onSizeChange, onPositionChange]);

  const handleMouseDown = useCallback(() => {
    onFocus(windowId);
  }, [windowId, onFocus]);

  const handleCloseClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onClose(windowId);
  }, [windowId, onClose]);

  // Get entity display information - use custom values if provided
  const entityType = node?.data?.entity_type || 'entity';
  const displayName = customTitle || 
                     node?.data?.name || 
                     node?.data?.entity_name || 
                     node?.data?.display_name ||
                     node?.data?.title ||
                     node?.data?.attributes?.name ||
                     node?.data?.attribute_map?.name ||
                     node?.data?.meta_data?.name ||
                     entityType ||
                     'Unnamed Entity';
  const displayIcon = customIcon || getEntityIcon(entityType);

  // For custom content windows, we don't need to check if node exists
  if (!customContent && (!node || !node.data)) {
    return null;
  }

  return (
    <Rnd
      position={position}
      size={size}
      onDragStop={handleDragStop}
      onResizeStop={handleResizeStop}
      onMouseDown={handleMouseDown}
      minWidth={300}
      minHeight={200}
      bounds="parent"
      dragHandleClassName="window-header"
      style={{ zIndex }}
      className="select-none"
    >
      <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg shadow-xl flex flex-col overflow-hidden">
        {/* Window Header */}
        <div className="window-header flex-shrink-0 bg-gray-800 border-b border-gray-700 px-3 py-2 flex items-center justify-between cursor-move">
          <div className="flex items-center gap-2 min-w-0">
            <div className="flex-shrink-0 text-sm">
              {displayIcon}
            </div>
            <div className="text-sm font-medium text-white truncate" title={displayName}>
              {displayName}
            </div>
            {!customContent && (
              <div className="text-xs text-gray-500 font-mono">
                {entityId.slice(0, 8)}...
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-1 flex-shrink-0">
            {/* Close Button */}
            <button
              onClick={handleCloseClick}
              className="w-6 h-6 rounded-full bg-red-600 hover:bg-red-500 text-white text-xs flex items-center justify-center transition-colors"
              title="Close window"
            >
              ×
            </button>
          </div>
        </div>

        {/* Window Content */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {customContent || (
            <EntityViewRenderer
              entityId={entityId}
              sendStrategyRequest={sendStrategyRequest}
              updateEntity={updateEntity}
            />
          )}
        </div>
      </div>
    </Rnd>
  );
};
