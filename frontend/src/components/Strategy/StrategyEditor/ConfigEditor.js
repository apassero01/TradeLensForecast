import React, { useState } from 'react';
import EntitySelector from '../inputs/EntitySelector';
import SequenceSetSelector from '../inputs/SequenceSetSelector';

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
        <textarea
          value={editedJson}
          onChange={handleJsonChange}
          className="w-full h-64 bg-gray-700/50 p-4 rounded-lg text-sm text-gray-300 font-mono 
                   focus:outline-none focus:ring-2 focus:ring-blue-500"
          spellCheck="false"
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