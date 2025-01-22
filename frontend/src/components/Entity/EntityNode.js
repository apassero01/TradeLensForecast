import React, { useState, useEffect, useCallback } from 'react';
import { Handle, Position } from 'reactflow';

import StrategyEditPanel from './Strategy/StrategyEditPanel';
import StrategyListModal from './Strategy/StrategyListModal';
import StrategyRequest from '../../utils/StrategyRequest';
import NodeStrategyPanel from './Strategy/NodeStrategyPanel';
import { EntityMenuTrigger } from './EntityMenu';
import BaseEntity from './BaseEntity';

const MIN_WIDTH = 250;
const MIN_HEIGHT = 100;
const EDITOR_WIDTH = 700;
const EDITOR_HEIGHT = 500;

function EntityNode({ data }) {
  const [localRequests, setLocalRequests] = useState(data.strategy_requests || []);
  
  // Show/hide the strategy list modal
  const [showStrategyList, setShowStrategyList] = useState(false);

  const [isEditing, setIsEditing] = useState(false);

  /**
   * editingRequest = {
   *   rawText: '...the JSON in the editor...',
   *   data: { strategy_name: '...', strategy_path: '...', etc. } // optional parsed object
   * }
   */
  const [editingRequest, setEditingRequest] = useState(null);

  // For resizing
  const [width, setWidth] = useState(MIN_WIDTH);
  const [height, setHeight] = useState(150);
  const [isResizing, setIsResizing] = useState(false);
  const [prevDims, setPrevDims] = useState({ w: MIN_WIDTH, h: 150 });

  // Expand node when editor is open
  useEffect(() => {
    if (editingRequest) {
      if (!isEditing) {
        setPrevDims({ w: width, h: height });
        setWidth(Math.max(width, EDITOR_WIDTH));
        setHeight(Math.max(height, EDITOR_HEIGHT));
        console.log('setting previous dims', prevDims)
        setIsEditing(true)
      }
    } else {
      // Revert to old dims
      setWidth(prevDims.w);
      setHeight(prevDims.h);
      console.log('reverting to old dims', prevDims)
      setIsEditing(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingRequest]);

  // Keep localRequests in sync if parent data changes
  useEffect(() => {
    if (
      Array.isArray(data.strategy_requests) &&
      JSON.stringify(data.strategy_requests) !== JSON.stringify(localRequests)
    ) {
      setLocalRequests(data.strategy_requests);
    }
  }, [data.strategy_requests]);

  // Called to show the strategy list
  const handleNewStrategy = () => setShowStrategyList(true);

  // Called when user picks a strategy from the list
  const handleSelectStrategy = (strategy) => {
    setShowStrategyList(false);

    // Build the initial request object
    const req = new StrategyRequest({
      ...strategy,
      target_entity_id: data.id, 
      param_config: strategy.config || {},
    }).toJSON();

    // Our state now includes both the raw text & the parsed data
    handleEditRequest(req);
  };

  // Called when user wants to edit an existing request
  const handleEditRequest = (request) => {
    if (!request) return;
    setEditingRequest({
      rawText: JSON.stringify(request, null, 2),
      data: request,
    });
  };

  // Called when user cancels the editor
  const handleCloseEditor = () => {
    setEditingRequest(null);
  };

  // Called as user types in the JSON Editor
  const handleUpdateEditingText = (newRawText) => {
    setEditingRequest((prev) => {
      if (!prev) return { rawText: newRawText, data: null };
      // Try to parse:
      try {
        const parsed = JSON.parse(newRawText);
        return {
          rawText: newRawText,
          data: parsed,
        };
      } catch {
        // If invalid JSON, keep old parsed data but update text
        return {
          ...prev,
          rawText: newRawText,
        };
      }
    });
  };

  // Called to actually execute the strategy
  const handleExecuteStrategy = () => {
    if (!editingRequest) return;

    try {
      const parsed = JSON.parse(editingRequest.rawText);
      console.log('Executing strategy:', parsed);
      console.log('editingRequest', editingRequest)

      // First close the editor
      setEditingRequest(null);

      // Then update the requests
      setLocalRequests((prev) => {
        const i = prev.findIndex(
          (r) =>
            r.strategy_name === parsed.strategy_name &&
            r.target_entity_id === parsed.target_entity_id
        );
        console.log('Found existing strategy at index:', i);
        if (i >= 0) {
          const copy = [...prev];
          copy[i] = parsed;
          return copy;
        }
        return [...prev, parsed];
      });

      // Finally execute the strategy
      data.onStrategyExecute?.(parsed);
    } catch (err) {
      console.error('Error parsing JSON on Execute:', err);
    }
  };

  // Resizing logic
  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);

    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = width;
    const startHeight = height;

    function onMouseMove(moveEvent) {
      moveEvent.preventDefault();
      moveEvent.stopPropagation();
      const newW = Math.max(MIN_WIDTH, startWidth + (moveEvent.clientX - startX));
      const newH = Math.max(MIN_HEIGHT, startHeight + (moveEvent.clientY - startY));
      setWidth(newW);
      setHeight(newH);
    }

    function onMouseUp() {
      setIsResizing(false);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [width, height]);

  return (
    <div
      style={{ width, height, minWidth: MIN_WIDTH, minHeight: MIN_HEIGHT }}
      className={`
        relative bg-gray-800 border border-gray-700 rounded-lg flex flex-col
        ${isResizing ? 'cursor-nwse-resize select-none' : 'cursor-grab active:cursor-grabbing'}
      `}
    >
      <Handle type="target" position={Position.Top} style={{ background: '#4b5563' }}/>
      <Handle type="source" position={Position.Bottom} style={{ background: '#4b5563' }}/>

      {editingRequest ? (
        <StrategyEditPanel
          editorText={editingRequest.rawText}
          onChangeText={handleUpdateEditingText}
          onClose={handleCloseEditor}
          onExecute={handleExecuteStrategy}
          onResize={handleResizeStart}
        />
      ) : (
        <EntityMenuTrigger
          onCopyId={() => navigator.clipboard.writeText(data.id)}
          onNewStrategy={handleNewStrategy}
        >
          <div className="w-full h-full flex flex-col">
            <div className="px-4 py-2 border-b border-gray-700">
              <div className="text-white font-medium">{data.entity_name}</div>
            </div>

            <BaseEntity entityData={data} />

            {localRequests.length > 0 && (
              <NodeStrategyPanel
                strategyRequests={localRequests}
                onExecute={data.onStrategyExecute}
                onEditRequest={handleEditRequest}
              />
            )}
          </div>
        </EntityMenuTrigger>
      )}

      {/* Resize handle */}
      <div
        className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize nodrag z-[100]"
        style={{
          background: 'linear-gradient(135deg, transparent 50%, #4b5563 50%)',
          borderBottomRightRadius: '0.5rem',
        }}
        onMouseDown={handleResizeStart}
      />

      <StrategyListModal
        show={showStrategyList}
        onClose={() => setShowStrategyList(false)}
        strategies={data.availableStrategies || {}}
        entityType={data.entity_type || 'entity'}
        onSelectStrategy={handleSelectStrategy}
        onRefresh={data.fetchAvailableStrategies}
      />
    </div>
  );
}

export default React.memo(EntityNode);