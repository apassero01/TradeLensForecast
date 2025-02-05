import React from 'react';

const ErrorDisplay = ({ message, onClose }) => {
  return (
    <div className="fixed top-16 left-1/2 transform -translate-x-1/2 
                bg-red-500/10 border border-red-500 text-red-500 
                px-6 py-4 rounded-lg shadow-lg z-50 
                max-w-md w-full mx-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="font-medium mb-1">Error</div>
          <div className="text-sm whitespace-pre-wrap">{message}</div>
        </div>
        {onClose && (
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onClose();
            }}
            className="ml-4 text-red-500 hover:text-red-700"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorDisplay; 