import React, { useState } from 'react';
import AceEditor from 'react-ace';
import EntitySelector from '../inputs/EntitySelector';
import SequenceSetSelector from '../inputs/SequenceSetSelector';
import ace from 'ace-builds';

// Import ace editor themes and modes
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/theme-monokai';
import 'ace-builds/src-noconflict/ext-language_tools';
import 'ace-builds/src-noconflict/worker-json';

// Configure ace paths
ace.config.set('basePath', '/ace-builds');
ace.config.set('modePath', '/ace-builds');
ace.config.set('themePath', '/ace-builds');
ace.config.set('workerPath', '/ace-builds');

const ConfigEditor = ({ 
  strategyRequest,
  onExecute
}) => {
  const [editedJson, setEditedJson] = useState(
    JSON.stringify(strategyRequest.toJSON(), null, 2)
  );
  const [error, setError] = useState(null);
  const [executionResult, setExecutionResult] = useState(null);

  console.log(strategyRequest);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const parsed = JSON.parse(editedJson);
      strategyRequest.param_config = parsed.param_config;
      const result = await onExecute(strategyRequest.toJSON());
      setExecutionResult(result.strategy_response);
      setError(null);
    } catch (error) {
      setError(error.message || 'Invalid JSON configuration');
    }
  };

  const handleJsonChange = (e) => {
    setEditedJson(e.target.value);
    setError(null);
    setExecutionResult(null);
  };

  const handleEntitySelect = (entityConfig) => {
    try {
      strategyRequest.setParameter('entity_class', entityConfig.entity_class);
      setEditedJson(JSON.stringify(strategyRequest.toJSON(), null, 2));
      setError(null);
    } catch (error) {
      setError('Error updating configuration');
    }
  };

  const handleSequenceSetSelect = (sequenceConfig) => {
    try {
      Object.entries(sequenceConfig).forEach(([key, value]) => {
        strategyRequest.setParameter(key, value);
      });
      setEditedJson(JSON.stringify(strategyRequest.toJSON(), null, 2));
      setError(null);
    } catch (error) {
      setError('Error updating configuration');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-medium text-white">
            Configure {strategyRequest.strategy_name}
          </h3>
        </div>
        {error && (
          <div className="flex items-center bg-red-500/10 text-red-500 px-4 py-2 rounded-lg">
            <span className="text-sm">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-3 text-red-500 hover:text-red-400"
            >
              ×
            </button>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Special input components for specific strategies */}
        {strategyRequest.strategy_name === 'CreateEntityStrategy' && (
          <EntitySelector 
            value={strategyRequest.param_config.entity_class || ''}
            onChange={handleEntitySelect}
          />
        )}
        
        {strategyRequest.strategy_name === 'GetSequenceSetsStrategy' && (
          <SequenceSetSelector
            value={strategyRequest.param_config}
            onChange={handleSequenceSetSelect}
          />
        )}
        
        {/* JSON Editor */}
        <AceEditor
          mode="json"
          theme="monokai"
          value={editedJson}
          onChange={setEditedJson}
          name="json-editor"
          editorProps={{ $blockScrolling: true }}
          setOptions={{
            showLineNumbers: true,
            tabSize: 2,
            useSoftTabs: true,
            showPrintMargin: false,
            highlightActiveLine: true,
            enableBasicAutocompletion: true,
            enableLiveAutocompletion: true,
            useWorker: true
          }}
          width="100%"
          height="32rem"
          className="rounded-lg"
          fontSize={14}
          onValidate={(annotations) => {
            if (annotations.length > 0) {
              console.log('JSON validation issues:', annotations);
            }
          }}
        />

        {/* Execution Result */}
        {executionResult && (
          <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
            <h4 className="text-sm font-medium text-green-400 mb-2">Strategy Response:</h4>
            <pre className="text-sm text-gray-300 font-mono overflow-auto">
              {JSON.stringify(executionResult, null, 2)}
            </pre>
          </div>
        )}

        {/* Execute Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600
                     transition-colors duration-150"
          >
            Execute Strategy
          </button>
        </div>
      </form>
    </div>
  );
};

export default ConfigEditor; 