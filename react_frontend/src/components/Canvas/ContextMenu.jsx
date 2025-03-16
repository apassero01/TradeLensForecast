import React, { useCallback, useState } from 'react';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
import { FaCopy, FaCheck } from 'react-icons/fa';

export default function ContextMenu({
  id,
  entityId,
  top,
  left,
  right,
  bottom,
  onClick,
}) {
  const { sendStrategyRequest } = useWebSocketConsumer();
  const [copied, setCopied] = useState(false);

  // Create shortened preview of entity ID
  const shortenedId = entityId?.length > 8 ? `${entityId.substring(0, 8)}...` : entityId;

  const duplicateNode = useCallback(() => {
    // Close the context menu
    onClick();
    
    // Send strategy request to duplicate the entity
    sendStrategyRequest({
      strategy_name: 'DuplicateEntityStrategy',
      target_entity_id: entityId,
      param_config: {},
      add_to_history: true,
      nested_requests: [],
    });
  }, [entityId, sendStrategyRequest, onClick]);

  const deleteNode = useCallback(() => {
    // Close the context menu
    onClick();
    
    // Send strategy request to delete the entity
    sendStrategyRequest({
      strategy_name: 'DeleteEntityStrategy',
      target_entity_id: entityId,
      param_config: {},
      add_to_history: true,
      nested_requests: [],
    });
  }, [entityId, sendStrategyRequest, onClick]);
  
  const copyIdToClipboard = useCallback((e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(entityId);
    
    // Show copied feedback briefly before closing
    setCopied(true);
    
    // Close the menu after a short delay to show the feedback
    setTimeout(() => {
      onClick();
    }, 300);
  }, [entityId, onClick]);

  return (
    <>
      {/* Transparent overlay to capture clicks outside the menu */}
      <div 
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 999,
        }}
        onClick={onClick}
        onContextMenu={(e) => {
          e.preventDefault();
          onClick();
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: top || undefined,
          left: left || undefined,
          right: right || undefined,
          bottom: bottom || undefined,
          zIndex: 1000,
          backgroundColor: '#1f2937',
          borderRadius: '4px',
          boxShadow: '0 2px 5px rgba(0,0,0,0.3)',
          border: '1px solid #374151',
        }}
        className="context-menu"
        onClick={(e) => e.stopPropagation()}
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onClick();
        }}
      >
        <div style={{ 
          margin: '0.5em', 
          color: 'white', 
          padding: '5px 10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <small title={entityId}>{shortenedId}</small>
          <div style={{ position: 'relative' }}>
            <button 
              onClick={copyIdToClipboard}
              style={{
                background: 'transparent',
                border: 'none',
                color: copied ? '#4ade80' : '#9ca3af',
                cursor: 'pointer',
                padding: '2px',
                marginLeft: '5px',
                display: 'flex',
                alignItems: 'center',
                transition: 'color 0.2s'
              }}
              title="Copy ID to clipboard"
            >
              {copied ? <FaCheck size={12} /> : <FaCopy size={12} />}
            </button>
            {copied && (
              <div style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                backgroundColor: '#4ade80',
                color: 'white',
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '10px',
                whiteSpace: 'nowrap',
                zIndex: 1001,
              }}>
                Copied!
              </div>
            )}
          </div>
        </div>
        <button 
          onClick={duplicateNode}
          style={{
            display: 'block',
            width: '100%',
            padding: '8px 10px',
            textAlign: 'left',
            backgroundColor: 'transparent',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            borderTop: '1px solid #374151',
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#374151'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Duplicate
        </button>
        <button 
          onClick={deleteNode}
          style={{
            display: 'block',
            width: '100%',
            padding: '8px 10px',
            textAlign: 'left',
            backgroundColor: 'transparent',
            border: 'none',
            color: '#ef4444',
            cursor: 'pointer',
            borderTop: '1px solid #374151',
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#374151'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Delete
        </button>
      </div>
    </>
  );
} 