import React, { useCallback, useState } from 'react';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
import { FaCopy, FaCheck } from 'react-icons/fa';
import ConfirmationModal from '../Modal/ConfirmationModal';

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
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

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

  const Visualization = useCallback(() => {
    // Close the context menu
    onClick();
    
    // Send strategy request to duplicate the entity
    sendStrategyRequest({
      strategy_name: 'CreateEntityStrategy',
      target_entity_id: entityId,
      param_config: {"entity_class": "shared_utils.entities.VisualizationEntity.VisualizationEntity"},
      add_to_history: false,
      nested_requests: [],
    });
  }, [entityId, sendStrategyRequest, onClick]);

  const copyEntityId = useCallback(() => {
    // Copy entity ID to clipboard
    navigator.clipboard.writeText(entityId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [entityId]);

  const deleteNode = useCallback(() => {
    // Show confirmation modal instead of immediate deletion
    setShowDeleteConfirm(true);
  }, []);

  const confirmDelete = useCallback(() => {
    // Perform deletion after confirmation
    // Close the context menu
    onClick();
    
    // Send strategy request to delete the entity
    sendStrategyRequest({
      strategy_name: 'RemoveEntityStrategy',
      param_config: {},
      target_entity_id: entityId,
      add_to_history: false,
      nested_requests: [],
    });
  }, [onClick, sendStrategyRequest, entityId]);

  return (
    <>
      {/* Overlay to capture clicks outside the menu */}
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
      
      {/* The context menu */}
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
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
          minWidth: '180px',
        }}
      >
        <div style={{ padding: '8px 10px', borderBottom: '1px solid #374151' }}>
          <div 
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <span style={{ fontSize: '12px', color: 'white' }}>{shortenedId}</span>
            <button
              onClick={copyEntityId}
              style={{
                background: 'none',
                border: 'none',
                color: '#9ca3af',
                cursor: 'pointer',
                padding: '2px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {copied ? <FaCheck size={12} color="#10b981" /> : <FaCopy size={12} />}
            </button>
            {copied && (
              <div style={{
                position: 'absolute',
                right: 10,
                top: '100%',
                backgroundColor: '#10b981',
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
          onClick={Visualization}
          style={{
            display: 'block',
            width: '100%',
            padding: '8px 10px',
            textAlign: 'left',
            backgroundColor: 'transparent',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#374151'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        >
          Visualization
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

      {/* Delete confirmation modal */}
      <ConfirmationModal
        isOpen={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={confirmDelete}
        title="Confirm Delete"
        message={`Are you sure you want to delete this entity (${shortenedId})?`}
        confirmText="Delete"
        cancelText="Cancel"
      />
    </>
  );
} 