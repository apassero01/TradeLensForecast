import React, { useState } from 'react';
import JSONEditorModal from '../utils/JSONEditorModal';

function ModelStageStrategyCard({ strategy, onSubmit, isSubmitted, onRemove }) {
  const [isEditing, setIsEditing] = useState(false);
  const [localConfig, setLocalConfig] = useState(strategy.config);

  const handleCardClick = () => {
    setIsEditing(true);
  };

  const handleSave = (newConfig) => {
    setLocalConfig(newConfig);
    setIsEditing(false);
  };

  const handleSubmit = async () => {
    await onSubmit({ ...strategy, config: localConfig });
  };

  // Customize the border color or any other styling as needed
  const borderColor = isSubmitted
    ? strategy.config.is_applied ? 'border-green-500' : 'border-red-500'
    : 'border-yellow-500';

  return (
    <div
      className={`relative flex flex-col items-center justify-between p-4 rounded-lg shadow-md cursor-pointer border-4 ${borderColor} ${isSubmitted ? 'bg-gray-800' : 'bg-yellow-700'}`}
      style={{ width: '150px', height: '150px' }}
      onClick={handleCardClick}
    >
      {/* Strategy Title */}
      <h3 className="text-sm font-semibold text-white text-center break-words max-w-xs mb-2">{strategy.name}</h3>

      {/* Submit Button */}
      <button
        className="absolute bottom-2 bg-green-600 text-white py-1 px-3 rounded-lg hover:bg-green-700"
        onClick={(e) => {
          e.stopPropagation();
          handleSubmit();
        }}
      >
        Submit
      </button>

      {/* Remove Button for Non-Submitted Strategies */}
      {!isSubmitted && onRemove && (
        <button
          className="absolute top-2 right-2 text-white bg-red-600 rounded-full w-6 h-6 flex items-center justify-center"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
        >
          X
        </button>
      )}

      {/* JSON Editor Modal */}
      {isEditing && (
        <JSONEditorModal
          initialConfig={localConfig}
          onSave={handleSave}
          onCancel={() => setIsEditing(false)}
        />
      )}
    </div>
  );
}

export default ModelStageStrategyCard;