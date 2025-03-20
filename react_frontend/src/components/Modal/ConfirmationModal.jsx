import React from 'react';

function ConfirmationModal({ 
  isOpen, 
  onClose, 
  onConfirm, 
  title = 'Confirm Action', 
  message = 'Are you sure you want to perform this action?',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  confirmColor = '#ef4444' // Red color for dangerous actions
}) {
  if (!isOpen) return null;
  
  return (
    <>
      {/* Modal backdrop */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          zIndex: 1100,
        }}
        onClick={onClose}
      />
      
      {/* Modal content */}
      <div
        style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: '#1f2937',
          borderRadius: '4px',
          padding: '16px',
          zIndex: 1101,
          minWidth: '300px',
          maxWidth: '90%',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          color: 'white',
        }}
      >
        <h3 style={{ 
          marginTop: 0, 
          fontSize: '18px',
          borderBottom: '1px solid #374151',
          paddingBottom: '8px'
        }}>
          {title}
        </h3>
        
        <div style={{ margin: '16px 0' }}>
          {message}
        </div>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'flex-end',
          gap: '8px',
          marginTop: '16px'
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              backgroundColor: 'transparent',
              border: '1px solid #4b5563',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer',
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#374151'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
          >
            {cancelText}
          </button>
          
          <button
            onClick={() => {
              onConfirm();
              onClose();
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: confirmColor,
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer',
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#dc2626'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = confirmColor}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </>
  );
}

export default ConfirmationModal; 