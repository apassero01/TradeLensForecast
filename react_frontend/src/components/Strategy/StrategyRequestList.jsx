import React, { useState, useEffect, useRef } from 'react';
import { DndContext, closestCenter } from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  verticalListSortingStrategy,
  useSortable
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { FaPlay, FaEdit, FaTimes } from 'react-icons/fa';
import { StrategyRequests } from '../../utils/StrategyRequestBuilder';
// Define the SortableItem component without memo for now
function SortableItem({ id, request, updateEntity, sendStrategyRequest, onRemoveRequest }) {
  // Use dnd-kit hook for the sortable item
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id });
  
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    display: 'flex',
    alignItems: 'center',
    marginBottom: '6px',
    backgroundColor: '#2d3748',
    padding: '6px 8px',
    borderRadius: '6px',
    userSelect: 'none',
    zIndex: 1010,
    position: 'relative'
  };

  // Use callbacks directly without useCallback for now
  const handlePlay = (e) => {
    e.stopPropagation();
    sendStrategyRequest(request);
  };

  const handleEdit = (e) => {
    e.stopPropagation();
    // updateEntity(request.entity_id, { hidden: false, selected: true, position: { x: e.clientX, y: e.clientY } });
    // sendStrategyRequest(
    //   StrategyRequests.setAttributes(request.entity_id, { position: { x: e.clientX, y: e.clientY }, hidden: false }),
    // );
    sendStrategyRequest(StrategyRequests.hideEntity(request.entity_id, false));
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onRemoveRequest(request.entity_id);
  };

  return (
    <div 
      ref={setNodeRef} 
      style={style}
    >
      <div 
        {...attributes} 
        {...listeners} 
        style={{ marginRight: '8px', padding: '0 4px', cursor: 'grab' }}
      >
        ≡
      </div>
      
      <button
        style={{
          backgroundColor: '#10B981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          padding: '6px',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
          pointerEvents: 'auto'
        }}
        onClick={handlePlay}
        title={request.strategy_name || 'Execute strategy'}
      >
        <FaPlay size={12} />
      </button>
      
      <div
        style={{
          flex: 1,
          marginLeft: '10px',
          color: 'white',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          fontSize: '14px',
          pointerEvents: 'auto'
        }}
      >
        {request.strategy_name || 'Strategy'}
      </div>
      
      <button
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: '#9CA3AF',
          padding: '4px',
          borderRadius: '4px',
          transition: 'all 0.2s ease',
          pointerEvents: 'auto'
        }}
        onClick={handleEdit}
      >
        <FaEdit size={14} />
      </button>
      
      <button
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: '#EF4444',
          padding: '4px',
          borderRadius: '4px',
          transition: 'all 0.2s ease',
          pointerEvents: 'auto'
        }}
        onClick={handleRemove}
        title="Remove request"
      >
        <FaTimes size={14} />
      </button>
    </div>
  );
}

// Main component without memo for now
function StrategyRequestList({ childrenRequests, updateEntity, sendStrategyRequest, onRemoveRequest, data }) {
  // Initialize local state to preserve order, defaulting to empty array.
  const [orderedRequests, setOrderedRequests] = useState([]);
  const containerRef = useRef(null);

  // Optimize the update logic to avoid unnecessary re-renders
  useEffect(() => {
    if (!childrenRequests) return;
    setOrderedRequests(childrenRequests);
  }, [childrenRequests]);

  // Handle drag end without useCallback for now
  const handleDragEnd = (event) => {
    const { active, over } = event;
    if (active && over && active.id !== over.id) {
      setOrderedRequests((items) => {
        const oldIndex = items.findIndex(item => String(item.entity_id) === active.id);
        const newIndex = items.findIndex(item => String(item.entity_id) === over.id);
        const newItems = arrayMove(items, oldIndex, newIndex);
        
        // Use the parentEntityId passed from EntityNodeBase
        const childIds = newItems.map(item => item.entity_id);
        
        // Send the strategy request for updating child list order
        sendStrategyRequest({
          strategy_name: 'UpdateChildrenStrategy',
          target_entity_id: data.entityId,
          param_config: { child_ids: [...new Set([...childIds, ...data.child_ids])] },
          add_to_history: false,
          nested_requests: [],
        });
        
        return newItems;
      });
    }
  };

  // Handle mouse events without useCallback for now
  const handleMouseEvent = (e) => {
    e.stopPropagation();
    
    const isInteractive = 
      e.target.tagName.toLowerCase() === 'button' || 
      e.target.tagName.toLowerCase() === 'input' ||
      e.target.closest('button') || 
      e.target.closest('input');
    
    if (!isInteractive) {
      e.preventDefault();
    }
  };

  // Handle remove request without useCallback for now
  const handleRemoveRequest = (entityId) => {
    onRemoveRequest && onRemoveRequest(entityId);
    setOrderedRequests(prev => prev.filter(item => item.entity_id !== entityId));
  };

  // If there are no requests, don't render anything
  if (!orderedRequests.length) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      style={{
        backgroundColor: 'rgba(30, 41, 59, 0.8)', // Lighter opacity background
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        padding: '8px',
        minWidth: '180px',
        maxWidth: '250px',
        border: '1px solid rgba(59, 76, 99, 0.7)', // Lighter opacity border
        marginLeft: '12px',
        position: 'relative',
        zIndex: 9999,
        pointerEvents: 'auto',
        overflow: 'auto',
        touchAction: 'none'
      }}
      onMouseDown={handleMouseEvent}
      onPointerDown={handleMouseEvent}
    >
      <DndContext 
        collisionDetection={closestCenter} 
        onDragEnd={handleDragEnd}
        modifiers={[]}
      >
        <SortableContext 
          items={orderedRequests.map(item => String(item.entity_id))} 
          strategy={verticalListSortingStrategy}
        >
          {orderedRequests.map((request) => (
            <SortableItem
              key={request.entity_id}
              id={String(request.entity_id)}
              request={request}
              updateEntity={updateEntity}
              sendStrategyRequest={sendStrategyRequest}
              onRemoveRequest={handleRemoveRequest}
            />
          ))}
        </SortableContext>
      </DndContext>
    </div>
  );
}

export default StrategyRequestList;