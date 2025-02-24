// src/components/StrategyManager.jsx
import React, { useState } from 'react';
import Modal from '../Modal';
import StrategyEditor from './StrategyEditor';

function StrategyManager() {
  const [showModal, setShowModal] = useState(false);
  const [existingRequest, setExistingRequest] = useState(null);
  const [entityType, setEntityType] = useState(null);

  // Called when the user wants to create or edit a strategy
  function openStrategyEditor(request, type) {
    setExistingRequest(request || null);
    setEntityType(type || null);
    setShowModal(true);
  }

  function closeModal() {
    setShowModal(false);
    // optionally reset existingRequest/entityType
  }

  return (
    <div>
      {/* If showModal, display the Modal with StrategyEditor */}
      {showModal && (
        <Modal onClose={closeModal}>
          <StrategyEditor
            existingRequest={existingRequest}
            entityType={entityType}
          />
          <div className="flex justify-end mt-4">
            <button 
              onClick={closeModal} 
              className="px-3 py-1 bg-gray-700 rounded hover:bg-gray-600"
            >
              Close
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

export default StrategyManager;