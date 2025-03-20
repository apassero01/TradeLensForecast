// src/components/StrategyEditor.jsx
import React, { useEffect, useState } from 'react';
import { useStrategyEditor } from '../../hooks/useStrategyEditor';
import StrategyList from './StrategyList';  
import EntitySelector from './RequestEditorComponents/EntitySelector';
import SequenceSetSelector from './RequestEditorComponents/SequenceSetSelector';
import Editor from '../Input/Editor';
// Possibly import other custom forms if you have them
import { entityApi } from '../../api/entityApi';

function StrategyEditor({ existingRequest, entityType }) {
  const {
    requestObj,
    setRequestObj,
    registry,
    registryLoading,
    registryError,
    executeStrategy,
  } = useStrategyEditor(existingRequest);

  const { strategy_name, param_config } = requestObj;

  const [localRegistry, setLocalRegistry] = useState([registry]);

  // Use a separate state for editor content
  const [editorContent, setEditorContent] = useState(
    JSON.stringify(requestObj, null, 2)
  );

  useEffect(() => {
    handleRefresh();
  }, []);

  // Add a useEffect to update editorContent when requestObj changes
  useEffect(() => {
    setEditorContent(JSON.stringify(requestObj, null, 2));
  }, [requestObj]);

  // 1. When the user picks a strategy from StrategyList
  function handleStrategySelect(selectedStrat) {
    console.log(selectedStrat);
    
    // Create the updated request object
    const updatedRequest = {
      ...requestObj,
      strategy_name: selectedStrat.name,
      param_config: selectedStrat.config.param_config ? selectedStrat.config.param_config : selectedStrat.config ,
    };
    
    // Update the request object state
    setRequestObj(updatedRequest);
    
    // Update the editor content to match
    setEditorContent(JSON.stringify(updatedRequest, null, 2));
  }

  // 2. If this is "CreateEntityStrategy," we show EntitySelector
  //    and store the selected entity_class in param_config.entity_class
  function handleEntityClassSelect(selected) {
    // Create the updated request object with the selected entity_class
    const updatedRequest = {
      ...requestObj,
      param_config: {
        ...requestObj.param_config,
        entity_class: selected.entity_class,
      },
    };
    
    // Update both states
    setRequestObj(updatedRequest);
    setEditorContent(JSON.stringify(updatedRequest, null, 2));
  }

  const handleSequenceSetSelect = (sequenceConfig) => {
    try {
      const parsed = JSON.parse(editorContent);
      const updated = {
        ...parsed,
        param_config: {
          ...parsed.param_config,
          X_features: sequenceConfig.X_features,
          y_features: sequenceConfig.y_features,
          model_set_configs: sequenceConfig.model_set_configs,
          dataset_type: sequenceConfig.dataset_type,
        }
      };

      setEditorContent(JSON.stringify(updated, null, 2));
      setRequestObj(updated);
    } catch (err) {
      console.error('Error updating sequence set:', err);
    }
  };

  // 3. For param_config editing, we can show a JSON Editor or custom forms
  function handleTextChange(newText) {
    setEditorContent(newText);
    
    try {
      const parsed = JSON.parse(newText);
      setRequestObj(parsed);
    } catch (err) {
      // Only log when debugging, not on every keystroke
      // console.error('Invalid JSON in param config');
    }
  }

  async function handleRefresh() {
    try {
      const newRegistry = await entityApi.getStrategyRegistry();
      const flattenedRegistry = Object.values(newRegistry).flat();
      setLocalRegistry(flattenedRegistry);
    } catch (error) {
      console.error('Failed to refresh strategy registry:', error);
      // Optionally set an error state here if you want to display it to the user
    }
  }

  function handleExecute() {
    executeStrategy();
  }

  if (registryLoading) return <div>Loading registry...</div>;
  if (registryError) return <div className="text-red-500">{registryError}</div>;

  return (
    <div className="flex flex-col h-full w-full">
      <h2 className="flex-none text-lg font-semibold">Strategy Editor</h2>
  
      <div className="flex-none w-full border border-gray-700 rounded mt-4">
        <StrategyList
          strategies={localRegistry}
          entityType={entityType}
          onSelect={handleStrategySelect}
          onRefresh={handleRefresh}
        />
      </div>
  
      {strategy_name === 'CreateEntityStrategy' && (
        <div className="flex-none w-full mt-4">
          <p className="text-sm text-gray-400 mb-2">Pick an entity class for creation:</p>
          <EntitySelector
            value={param_config.entity_class || ''}
            onChange={handleEntityClassSelect}
          />
        </div>
      )}

      {strategy_name === 'GetSequenceSetsStrategy' && (
        <SequenceSetSelector
          value={param_config || {}}
          onChange={handleSequenceSetSelect}
        />
      )}
  
      <div className="flex-grow min-h-0 w-full mt-4">
        <p className="text-sm text-gray-400 mb-2">Param Config (JSON)</p>
        <div className="h-full w-full">
          <Editor
            visualization={{ 
              data: editorContent, 
              config: { type: 'json' } 
            }}
            onChange={handleTextChange}
          />
        </div>
      </div>

      <div className="flex-none w-full mt-4">
        <button
          onClick={handleExecute}
          className="mt-2 px-2 py-1 bg-gray-700 text-gray-200 rounded hover:bg-gray-600"
        >
          Execute
        </button>
      </div>
    </div>
  );
}

export default StrategyEditor;