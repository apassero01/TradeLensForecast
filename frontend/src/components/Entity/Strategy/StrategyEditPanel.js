import React, { useEffect, useState } from 'react';
import Editor from '../../Visualization/Editor';
import EntitySelector from './RequestEditorComponents/EntitySelector';
import SequenceSetSelector from './RequestEditorComponents/SequenceSetSelector';
import StrategyRequest from '../../../utils/StrategyRequest';

function StrategyEditPanel({
  editorText,      // The entire JSON string from parent
  onChangeText,    // Called on every keystroke
  onClose,
  onExecute,
  onResize
}) {
  const [error, setError] = useState(null);

  // Close on Esc
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  const handleEditorChange = (newValue) => {
    onChangeText(newValue); 
    setError(null);
  };

  // Helper to parse the current text to see what strategy_name is etc.
  let parsed;
  let strategyName = '';
  try {
    parsed = JSON.parse(editorText);
    strategyName = parsed.strategy_name || '';
  } catch (err) {
    // No big deal if invalid
  }

  // Example: If you need special selectors:
  const handleEntitySelect = (entityConfig) => {
    try {
      const parsed = JSON.parse(editorText);
      const updated = new StrategyRequest({
        name: parsed.strategy_name,
        path: parsed.strategy_path,
        config: {
          ...parsed.param_config,
          entity_class: entityConfig.entity_class,
        },
        nested_requests: parsed.nested_requests,
        add_to_history: parsed.add_to_history,
        target_entity_id: parsed.target_entity_id,
        entity_id: parsed.entity_id,
      }).toJSON();

      onChangeText(JSON.stringify(updated, null, 2));
      setError(null);
    } catch (err) {
      setError('Error updating entity');
    }
  };

  const handleSequenceSetSelect = (sequenceConfig) => {
    try {
      const parsed = JSON.parse(editorText);
      const updated = {
        ...parsed,
        param_config: {
          ...parsed.param_config,
          ...sequenceConfig
        }
      };

      onChangeText(JSON.stringify(updated, null, 2));
      setError(null);
    } catch (err) {
      setError('Error updating sequence set');
      console.error('Error updating sequence set:', err);
    }
  };

  const handleMouseDown = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // If near bottom-right corner, let parent handle resize
    if (x > rect.width - 20 && y > rect.height - 20) {
      onResize?.(e);
    }
  };

  return (
    <div 
      className="absolute inset-0 flex flex-col bg-gray-900 rounded-lg overflow-hidden z-10"
      onMouseDown={handleMouseDown}
    >
      {/* Header */}
      <div className="flex-none p-4 flex items-center justify-between border-b border-gray-700">
        <div>
          <h2 className="text-sm text-white font-medium">Edit Strategy Request</h2>
          <span className="text-xs text-gray-400">
            {strategyName}
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white px-3 py-1"
        >
          Cancel
        </button>
      </div>

      {/* Special UI based on strategyName */}
      <div className="flex-none p-4 space-y-4 border-b border-gray-700">
        {strategyName === 'CreateEntityStrategy' && (
          <EntitySelector 
            value={parsed?.param_config?.entity_class || ''}
            onChange={handleEntitySelect}
          />
        )}
        
        {strategyName === 'GetSequenceSetsStrategy' && (
          <SequenceSetSelector
            value={parsed?.param_config || {}}
            onChange={handleSequenceSetSelect}
          />
        )}

        {error && (
          <div className="bg-red-500/10 text-red-500 px-4 py-2 rounded-lg text-sm">
            {error}
          </div>
        )}
      </div>

      {/* JSON Editor */}
      <div className="flex-grow min-h-0 relative">
        <div className="absolute inset-0">
          <Editor
            visualization={{
              data: editorText,
              config: {
                type: 'json',
                title: strategyName,
                readOnly: false
              }
            }}
            onChange={handleEditorChange}
          />
        </div>
      </div>

      {/* Footer */}
      <div className="flex-none p-4 border-t border-gray-700 flex justify-end">
        <button
          onClick={onExecute}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-500 text-sm"
        >
          Execute Strategy
        </button>
      </div>
    </div>
  );
}

export default StrategyEditPanel;