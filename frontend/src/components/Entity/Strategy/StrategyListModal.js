// StrategyListModal.js
import React from 'react';
import StrategyList from './StrategyList';

function StrategyListModal({
  show,
  onClose,
  strategies,
  entityType,
  onSelectStrategy,
  onRefresh
}) {
  if (!show) return null;

  const handleBackdropClick = () => onClose();

  return (
    <div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
      onClick={handleBackdropClick}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded p-4 w-[400px] h-[500px]"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-white text-lg mb-2">Select a Strategy</h2>
        <StrategyList
          strategies={strategies}
          entityType={entityType}
          onSelect={onSelectStrategy}
          onRefresh={onRefresh}
        />
      </div>
    </div>
  );
}

export default StrategyListModal;