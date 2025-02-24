// src/components/StrategyEditor.jsx
import React from 'react';
import { useStrategyEditor } from '../../hooks/useStrategyEditor';
import StrategyList from './StrategyList';
import EntitySelector from './RequestEditorComponents/EntitySelector';
import Editor from '../Input/Editor';
// Possibly import other custom forms if you have them

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

  // 1. When the user picks a strategy from StrategyList
  function handleStrategySelect(selectedStrat) {
    setRequestObj((prev) => ({
      ...prev,
      strategy_name: selectedStrat.name,
    }));
  }

  // 2. If this is "CreateEntityStrategy," we show EntitySelector
  //    and store the selected entity_class in param_config.entity_class
  function handleEntityClassSelect(selected) {
    // Suppose selected = { entity_class: 'MyCustomClass', ... }
    setRequestObj((prev) => ({
      ...prev,
      param_config: {
        ...prev.param_config,
        entity_class: selected.entity_class,
      },
    }));
  }

  // 3. For param_config editing, we can show a JSON Editor or custom forms
  function handleParamConfigChange(newJson) {
    try {
      const parsed = JSON.parse(newJson);
      setRequestObj((prev) => ({
        ...prev,
        parsed,
      }));
    } catch (err) {
      console.error('Invalid JSON in param config');
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
          strategies={registry}
          entityType={entityType}
          onSelect={handleStrategySelect}
          onRefresh={() => {/* optional refetch if needed */}}
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
  
      <div className="flex-grow min-h-0 w-full mt-4">
        <p className="text-sm text-gray-400 mb-2">Param Config (JSON)</p>
        <div className="h-full w-full">
          <Editor
            visualization={{ data: JSON.stringify(requestObj, null, 2), config: { type: 'json' } }}
            onChange={handleParamConfigChange}
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