// src/components/Canvas/InputEntity.jsx
import React, { memo, useState, useEffect } from 'react';
import EntityNodeBase from './EntityNodeBase';
import { useRecoilValue } from 'recoil';
import { strategyRequestChildrenSelector } from '../../../state/entitiesSelectors';

function InputEntity({ data, updateEntity }) {
  const [text, setText] = useState(data.visualization || '');
  const [isSubmitting, setIsSubmitting] = useState(false);
  // Get strategy requests associated with this entity
  const strategyRequests = useRecoilValue(strategyRequestChildrenSelector(data.entityId));
  
  useEffect(() => {
    console.log('Strategy requests for this input:', strategyRequests);
  }, [strategyRequests]);

  const handleChange = (e) => {
    const newValue = e.target.value;
    // Update both the local state and the parent's local field.
    setText(newValue);
  };

  const handleKeyDown = (e) => {
    // Submit on Enter without Shift key (Shift+Enter allows for newlines)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      e.target.form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    }
  };

  return (
    <EntityNodeBase data={data} updateEntity={updateEntity}>
      {({ sendStrategyRequest }) => {
                  
        const execute_request = {
          strategy_name: 'ExecuteRequestChildren',
          param_config: {
          },
          target_entity_id: data.entityId,
          add_to_history: false,
          nested_requests: [],
        }
        
        const handleSubmit = async (e) => {
          e.preventDefault();
          console.log('Submitting with text:', text);
          
          // Activate submission feedback
          setIsSubmitting(true);
          
          await sendStrategyRequest({
            strategy_name: 'SetAttributesStrategy',
            param_config: {
              attribute_map: {
                'user_input': text,
              }
            },
            target_entity_id: data.parent_ids[0] ? data.parent_ids[0] : data.entityId,
            add_to_history: false,
            nested_requests: [],
          });
          
          // Execute all the strategy requests
          await sendStrategyRequest(execute_request);
          
          // Clear the text input
          setText('');
          
          // Remove the submission effect after a short delay
          setTimeout(() => setIsSubmitting(false), 800);
        };
        
        return (
          <form onSubmit={handleSubmit} className="w-full nodrag">
            <textarea
              value={text}
              onChange={(e) => handleChange(e)}
              onKeyDown={(e) => handleKeyDown(e)}
              placeholder="Enter text..."
              className={`w-full h-24 p-2 text-white rounded nodrag transition-all duration-200 ${
                isSubmitting 
                  ? 'bg-green-900 border-4 border-green-400 shadow-[0_0_15px_rgba(74,222,128,0.8)]' 
                  : 'bg-gray-700 border-2 border-transparent'
              }`}
              style={{ resize: 'none' }}
            />
            <button
              type="submit"
              className="mt-2 w-full px-2 py-1 bg-gray-600 hover:bg-gray-500 text-white rounded nodrag"
            >
              {strategyRequests.length > 0 
                ? `Submit & Execute ${strategyRequests.length} Strategies` 
                : 'Submit'}
            </button>
            
            {strategyRequests.length > 0 && (
              <div className="mt-2 text-xs text-gray-400">
                <p>Will execute the following strategies:</p>
                <ul className="list-disc pl-4 mt-1">
                  {strategyRequests.map((strategy, index) => (
                    <li key={index}>
                      {strategy.strategy_name || 'Unnamed Strategy'}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </form>
        );
      }}
    </EntityNodeBase>
  );
}

export default memo(InputEntity);