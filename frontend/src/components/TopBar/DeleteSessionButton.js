import React, { useState } from 'react';

const DeleteSessionButton = ({ onDeleteSession, isLoading }) => {
  const [showConfirm, setShowConfirm] = useState(false);

  const handleDelete = () => {
    onDeleteSession();
    setShowConfirm(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowConfirm(true)}
        disabled={isLoading}
        className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 
                 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Delete Session
      </button>

      {showConfirm && (
        <div className="absolute right-0 top-full mt-2 p-4 bg-gray-800 border border-gray-700 
                      rounded-lg shadow-lg z-50 w-64">
          <p className="text-white text-sm mb-4">
            Are you sure you want to delete this session? This action cannot be undone.
          </p>
          <div className="flex justify-end space-x-2">
            <button
              onClick={() => setShowConfirm(false)}
              className="px-3 py-1.5 text-sm bg-gray-700 text-white rounded hover:bg-gray-600"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700"
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DeleteSessionButton; 