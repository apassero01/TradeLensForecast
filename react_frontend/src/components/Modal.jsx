// src/components/Modal.jsx
import React from 'react';

function Modal({ onClose, children }) {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50">
      {/* Background overlay */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-50" 
        onClick={onClose}
      />
      {/* Modal content container */}
      <div className="relative bg-gray-800 text-gray-200 p-4 rounded shadow-lg z-10 w-[600px] max-w-full">
        {children}
      </div>
    </div>
  );
}

export default Modal;