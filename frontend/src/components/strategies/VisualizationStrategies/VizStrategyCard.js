// VizStrategyCard.js
import React, { useState, useEffect, useRef } from 'react';
import JSONEditorModal from '../utils/JSONEditorModal';

function VizStrategyCard({ strategy, onSubmit }) {
  const [isEditing, setIsEditing] = useState(false);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [localConfig, setLocalConfig] = useState(strategy.config);
  const cardRef = useRef(null);

  // Left-click handler: submits the strategy
  const handleLeftClick = (e) => {
    if (showContextMenu) {
      // If context menu is open, don't submit
      return;
    }
    e.preventDefault();
    onSubmit({ ...strategy, config: localConfig });
  };

  // Right-click handler: opens the context menu
  const handleRightClick = (e) => {
    e.preventDefault();
    setShowContextMenu(true);
  };

  // Handle selection of context menu options
  const handleContextMenuOptionClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsEditing(true);
    setShowContextMenu(false);
  };

  // Close the context menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (cardRef.current && !cardRef.current.contains(event.target)) {
        setShowContextMenu(false);
      }
    };

    if (showContextMenu) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showContextMenu]);

  // Handle saving the new configuration from the JSON editor
  const handleSave = (newConfig) => {
    setLocalConfig(newConfig);
    setIsEditing(false);
  };

  return (
    <div
      ref={cardRef}
      className="relative w-8 h-8 bg-blue-600 rounded flex items-center justify-center cursor-pointer"
      onClick={handleLeftClick}
      onContextMenu={handleRightClick}
      title={strategy.name}
    >
      <span className="text-white text-xs font-bold">
        {strategy.name.charAt(0).toUpperCase()}
      </span>

      {/* Custom Context Menu */}
      {showContextMenu && (
        <ul
          className="absolute bg-gray-800 text-white py-1 rounded shadow-lg z-10"
          style={{ top: '100%', left: 0 }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <li
            className="px-2 py-1 hover:bg-gray-700 cursor-pointer"
            onClick={handleContextMenuOptionClick}
          >
            Edit
          </li>
        </ul>
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

export default VizStrategyCard;