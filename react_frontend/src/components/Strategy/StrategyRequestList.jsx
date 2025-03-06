import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useReactFlow } from '@xyflow/react';
import { FaPlay, FaEdit, FaTimes } from 'react-icons/fa'; // Added FaTimes for "X"

function StrategyRequestList({ childrenRequests, updateEntity, sendStrategyRequest, onRemoveRequest }) {
  const [orderedRequests, setOrderedRequests] = useState([]);
  const [dragState, setDragState] = useState({
    dragging: false,
    index: null,
    startY: 0,
    deltaY: 0,
  });
  const containerRef = useRef(null);
  const itemRefs = useRef([]);
  const { transform } = useReactFlow();
  const zoom = transform ? transform[2] : 1;

  // Initialize orderedRequests with isSelected property
  useEffect(() => {
    if (childrenRequests && childrenRequests.length > 0) {
      const requestsWithSelection = childrenRequests.map((req) => ({
        ...req,
        isSelected: false, // Add isSelected to each request
      }));
      setOrderedRequests(requestsWithSelection);
      itemRefs.current = requestsWithSelection.map(() => null);
    }
  }, [childrenRequests]);

  const handlePointerMove = useCallback(
    (e) => {
      const selectedRequest = orderedRequests.find((req) => req.isSelected);
      if (!dragState.dragging || dragState.index === null || !selectedRequest) {
        console.log('No selected request or not dragging, exiting move');
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      console.log('Moving, deltaY:', dragState.deltaY); // Debug move

      const currentY = e.clientY;
      const deltaY = (currentY - dragState.startY) / zoom;
      const containerRect = containerRef.current.getBoundingClientRect();
      const relativeY = (currentY - containerRect.top) / zoom;

      let newIndex = dragState.index;
      itemRefs.current.forEach((itemEl, i) => {
        if (i === dragState.index || !itemEl) return;
        const itemRect = itemEl.getBoundingClientRect();
        const itemTop = (itemRect.top - containerRect.top) / zoom;
        const itemBottom = (itemRect.bottom - containerRect.top) / zoom;

        if (relativeY >= itemTop && relativeY <= itemBottom) {
          newIndex = i;
        }
      });

      if (newIndex !== dragState.index) {
        console.log(`Reordering from ${dragState.index} to ${newIndex}`);
        const newOrdered = [...orderedRequests];
        const [movedItem] = newOrdered.splice(dragState.index, 1);
        newOrdered.splice(newIndex, 0, movedItem);
        setOrderedRequests(newOrdered);
        setDragState((prev) => ({ ...prev, index: newIndex, deltaY }));
      } else {
        setDragState((prev) => ({ ...prev, deltaY }));
      }
    },
    [dragState, orderedRequests, zoom]
  );

  const handlePointerUp = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      console.log('Pointer up, ending drag'); // Debug end
      if (dragState.dragging) {
        // Deselect the request and reset drag state
        const newOrdered = orderedRequests.map((req, i) =>
          i === dragState.index ? { ...req, isSelected: false } : req
        );
        setOrderedRequests(newOrdered);
        setDragState({ dragging: false, index: null, startY: 0, deltaY: 0 });
        window.removeEventListener('pointermove', handlePointerMove, { capture: true });
        window.removeEventListener('pointerup', handlePointerUp, { capture: true });
        document.body.style.userSelect = '';
      }
    },
    [dragState.dragging, dragState.index, orderedRequests, handlePointerMove]
  );

  const handlePointerDown = useCallback(
    (e, index) => {
      if (!e.target.classList.contains('drag-handle')) return;
      e.preventDefault();
      e.stopPropagation();
      console.log('Pointer down, starting drag at index:', index); // Debug start

      // Select the request and start dragging
      const newOrdered = orderedRequests.map((req, i) =>
        i === index ? { ...req, isSelected: true } : { ...req, isSelected: false }
      );
      setOrderedRequests(newOrdered);
      setDragState({ dragging: true, index, startY: e.clientY, deltaY: 0 });
      window.addEventListener('pointermove', handlePointerMove, { capture: true });
      window.addEventListener('pointerup', handlePointerUp, { capture: true });
      document.body.style.userSelect = 'none';
    },
    [orderedRequests, handlePointerMove, handlePointerUp]
  );

  const handleRemoveRequest = useCallback(
    (e, index) => {
      e.stopPropagation(); // Prevent bubbling to parent elements
      console.log('Removing request at index:', index); // Debug removal
      const request = orderedRequests[index];
      if (onRemoveRequest && request) {
        onRemoveRequest(request.entity_id); // Call the parent-provided removal function
        const newOrdered = orderedRequests.filter((_, i) => i !== index); // Remove locally
        setOrderedRequests(newOrdered);
        itemRefs.current = newOrdered.map(() => null); // Update refs
      }
    },
    [orderedRequests, onRemoveRequest]
  );

  if (!orderedRequests || orderedRequests.length === 0) return null;

  const getItemStyle = (index) => ({
    display: 'flex',
    alignItems: 'center',
    marginBottom: '6px',
    backgroundColor: orderedRequests[index]?.isSelected ? '#4a5568' : '#2d3748', // Highlight selected
    padding: '6px 8px',
    borderRadius: '6px',
    transition: dragState.dragging && dragState.index === index ? 'none' : 'all 0.2s ease',
    cursor: 'default',
    opacity: dragState.dragging && dragState.index === index ? 0.7 : 1,
    position: dragState.dragging && dragState.index === index ? 'relative' : 'static',
    transform: dragState.dragging && dragState.index === index ? `translateY(${dragState.deltaY}px)` : 'none',
    boxShadow: dragState.dragging && dragState.index === index ? '0 4px 8px rgba(0, 0, 0, 0.3)' : 'none',
    zIndex: dragState.dragging && dragState.index === index ? 10 : 1,
  });

  return (
    <div
      ref={containerRef}
      style={{
        backgroundColor: '#1e293b',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
        padding: '8px',
        minWidth: '180px',
        maxWidth: '250px',
        border: '1px solid #3b4c63',
        marginLeft: '12px',
        position: 'relative',
        zIndex: 1000,
        pointerEvents: 'auto',
      }}
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <div style={{ color: 'white', marginBottom: '8px', fontWeight: 'bold', fontSize: '14px' }}>
        Strategy Requests
      </div>
      {orderedRequests.map((request, index) => (
        <div
          key={request.entity_id || request.id || index}
          ref={(el) => (itemRefs.current[index] = el)}
          style={getItemStyle(index)}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            className="drag-handle"
            style={{
              marginRight: '8px',
              cursor: 'move',
              touchAction: 'none',
              userSelect: 'none',
              padding: '0 4px',
            }}
            onPointerDown={(e) => handlePointerDown(e, index)}
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
            }}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              console.log('Play button clicked');
              sendStrategyRequest(request);
            }}
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
            }}
          >
            {request.label || request.strategy_name || 'Strategy'}
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
            }}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              console.log('Edit button clicked');
              updateEntity(request.entity_id, {
                hidden: false,
                width: 695,
                height: 426,
              });
            }}
          >
            <FaEdit size={14} />
          </button>
          <button
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#EF4444', // Red color for remove button
              padding: '4px',
              borderRadius: '4px',
              transition: 'all 0.2s ease',
            }}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => handleRemoveRequest(e, index)}
            title="Remove request"
          >
            <FaTimes size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

export default StrategyRequestList;