// src/components/StrategyEditor.jsx
import React, { useEffect, useState } from 'react';
import { useStrategyEditor } from '../../hooks/useStrategyEditor';
import StrategyList from './StrategyList';  
import EntitySelector from './RequestEditorComponents/EntitySelector';
import SequenceSetSelector from './RequestEditorComponents/SequenceSetSelector';
import Editor from '../Input/Editor';
// Possibly import other custom forms if you have them
import { entityApi } from '../../api/entityApi';

function StrategyEditor({ existingRequest, entityType, updateEntity, sendStrategyRequest, isLoading, setIsLoading }) {
  const {
    requestObj,
    setRequestObj,
    registry,
    registryLoading,
    registryError,
    executeStrategy,
    refresh,
  } = useStrategyEditor(existingRequest);

  const { strategy_name, param_config } = requestObj;

  const [saveRequest, setSaveRequest] = useState(false);

  // Use a separate state for editor content
  const [editorContent, setEditorContent] = useState(
    JSON.stringify(requestObj, null, 2)
  );

  // // Add a useEffect to update editorContent when requestObj changes
  // useEffect(() => {
  //   setEditorContent(JSON.stringify(requestObj, null, 2));
  // }, [requestObj]);

  // Update the useEffect to sync the saveRequest state with requestObj
  useEffect(() => {
    const updatedRequest = {
      ...requestObj,
      add_to_history: saveRequest
    };
    setRequestObj(updatedRequest);
  }, [saveRequest]);

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

  // async function handleRefresh() {
  //   try {
  //     const newRegistry = await entityApi.getStrategyRegistry();
  //     const flattenedRegistry = Object.values(newRegistry).flat();
  //     setRegistry(flattenedRegistry);
  //   } catch (error) {
  //     console.error('Failed to refresh strategy registry:', error);
  //     // Optionally set an error state here if you want to display it to the user
  //   }
  // }

  function handleExecute() {
    executeStrategy();
  }

  if (registryLoading) return <div>Loading registry...</div>;
  if (registryError) return <div className="text-red-500">{registryError}</div>;

  return (
    <div className="flex flex-col h-full w-full relative">
      <h2 className="flex-none text-lg font-semibold">Strategy Editor</h2>
  
      <div className="flex-none w-full border border-gray-700 rounded mt-4">
        <StrategyList
          strategies={registry || []}
          entityType={entityType}
          onSelect={handleStrategySelect}
          onRefresh={refresh}
          selectedStrategy={strategy_name}
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
  
      <div className="flex-grow min-h-0 w-full mt-4 mb-20">
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

      <div className="fixed bottom-4 left-4 py-3 px-4 bg-gray-800 border border-gray-700 flex items-center space-x-4 rounded-md shadow-lg">
        <button
          onClick={handleExecute}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500 font-medium"
        >
          Execute
        </button>
        <div className="flex items-center">
          <input
            type="checkbox"
            id="saveRequest"
            checked={saveRequest}
            onChange={(e) => setSaveRequest(e.target.checked)}
            className="mr-2 h-4 w-4"
          />
          <label htmlFor="saveRequest" className="text-gray-300 text-sm">
            Save Request
          </label>
        </div>
      </div>
    </div>
  );
}

export default StrategyEditor;