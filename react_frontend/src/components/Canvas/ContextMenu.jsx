import React, { useCallback, useState, useRef, useEffect } from 'react';
import { useWebSocketConsumer } from '../../hooks/useWebSocketConsumer';
import { FaCopy, FaCheck } from 'react-icons/fa';
import ConfirmationModal from '../Modal/ConfirmationModal';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector, nodeSelectorFamily } from '../../state/entitiesSelectors';
import { EntityTypes } from '../Canvas/Entity/EntityEnum';
import VisibilityToggleIcon from '../common/VisibilityToggleIcon';
import { StrategyRequests } from '../../utils/StrategyRequestBuilder';
import { useReactFlow } from '@xyflow/react';
import useUpdateFlowNodes from '../../hooks/useUpdateFlowNodes';
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
  const [showViewsSubmenu, setShowViewsSubmenu] = useState(false);
  const viewsMenuRef = useRef(null);
  const reactFlowInstance = useReactFlow();
  const shortenedId = entityId?.length > 8 ? `${entityId.substring(0, 8)}...` : entityId;

  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: entityId, type: EntityTypes.VIEW }));
  const updateEntity = useUpdateFlowNodes(reactFlowInstance);
  const entity = useRecoilValue(nodeSelectorFamily(entityId));

  const handleToggleVisibility = useCallback((viewId, isHidden) => {
    const hideRequest = StrategyRequests.hideEntity(viewId, !isHidden);
    sendStrategyRequest(hideRequest);
    updateEntity(viewId, { hidden: !isHidden });
  }, [sendStrategyRequest, updateEntity]);

  // Close views submenu when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (viewsMenuRef.current && !viewsMenuRef.current.contains(event.target)) {
        setShowViewsSubmenu(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

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

  const newView = useCallback(() => {
    // Close the context menu
    onClick();

    const create_request = StrategyRequests.createEntity(entityId, "shared_utils.entities.view_entity.ViewEntity.ViewEntity", { hidden: false, width: 350, height: 350 });
    sendStrategyRequest([create_request]);
  }, [entityId, sendStrategyRequest, onClick, updateEntity]);

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

  const hideEntity = useCallback(() => {
    onClick(); // Close context menu

    // Strategy request to set hidden to false (unhide)
    const unhideRequest = StrategyRequests.hideEntity(entityId, true);
    sendStrategyRequest(unhideRequest);

  }, [onClick, entityId, sendStrategyRequest]);

  const unhideAllChildren = useCallback(() => {
    // TODO: Implement unhide all children functionality
    const unhideRequest = StrategyRequests.hideEntity(entityId, false);
    sendStrategyRequest(unhideRequest);
  }, [entityId, sendStrategyRequest]);

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
          onClick={newView}
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
          New View
        </button>

        {/* Views submenu button - only show if there are views */}
        {viewChildren && viewChildren.length > 0 && (
          <div
            style={{ position: 'relative' }}
            onMouseEnter={() => setShowViewsSubmenu(true)}
            onMouseLeave={() => setShowViewsSubmenu(false)}
          >
            <button
              style={{
                display: 'block',
                width: '100%',
                padding: '8px 10px',
                textAlign: 'left',
                backgroundColor: showViewsSubmenu ? '#374151' : 'transparent',
                border: 'none',
                color: 'white',
                cursor: 'pointer',
                borderTop: '1px solid #374151',
                position: 'relative',
              }}
            >
              Views
              <span style={{ position: 'absolute', right: '10px' }}>▶</span>
            </button>

            {/* Views submenu */}
            {showViewsSubmenu && (
              <div
                ref={viewsMenuRef}
                style={{
                  position: 'absolute',
                  left: '100%',
                  top: 0,
                  backgroundColor: '#1f2937',
                  borderRadius: '4px',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
                  minWidth: '180px',
                  zIndex: 1001,
                }}
              >
                {viewChildren.map((view) => {
                  const viewName = view.data.view_component_type ||
                    `View ${view.id.substring(0, 6)}`;
                  const isHidden = !!view.data.hidden;

                  return (
                    <div
                      key={view.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        padding: '8px 10px',
                        color: 'white',
                        borderBottom: '1px solid #374151',
                      }}
                      onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#374151'}
                      onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <div style={{ width: '40px', display: 'flex', justifyContent: 'center' }}>
                        <VisibilityToggleIcon
                          isHidden={isHidden}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleVisibility(view.id, isHidden);
                          }}
                        />
                      </div>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {viewName}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <button
          onClick={hideEntity}
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
          Hide
        </button>

        <button
          onClick={unhideAllChildren}
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
          Unhide all children
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