// src/components/Canvas/InputEntity.jsx
import React, { memo, useState, useEffect } from 'react';
import EntityNodeBase from './EntityNodeBase';
import { useRecoilValue } from 'recoil';
import { strategyRequestChildrenSelector } from '../../../state/entitiesSelectors';

function InputEntity({ data }) {
  const [text, setText] = useState(data.visualization || '');
  // Get strategy requests associated with this entity
  const strategyRequests = useRecoilValue(strategyRequestChildrenSelector(data.entityId));
  
  useEffect(() => {
    console.log('Strategy requests for this input:', strategyRequests);
  }, [strategyRequests]);

  const handleChange = (e, updateLocalField) => {
    const newValue = e.target.value;
    // Update both the local state and the parent's local field.
    setText(newValue);
    updateLocalField('visualization', newValue);
  };

  return (
    <EntityNodeBase data={data}>
      {({ updateLocalField, sendStrategyRequest }) => {
        // This function executes all attached strategy requests
        const executeStrategyRequests = async () => {
          console.log('Executing strategy requests:', strategyRequests);
          
          // First set the user input value
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
          
          // Then execute each strategy in the StrategyRequestList
          for (const strategy of strategyRequests) {
              
              try {
                await sendStrategyRequest(strategy);
                console.log('Strategy executed successfully');
              } catch (error) {
                console.error('Failed to execute strategy:', error);
              }
            }
        };
        
        const handleSubmit = async (e) => {
          e.preventDefault();
          console.log('Submitting with text:', text);
          
          // Execute all the strategy requests
          await executeStrategyRequests();
        };
        
        return (
          <form onSubmit={handleSubmit} className="w-full nodrag">
            <textarea
              value={text}
              onChange={(e) => handleChange(e, updateLocalField)}
              placeholder="Enter text..."
              className="w-full h-24 p-2 bg-gray-700 text-white rounded nodrag"
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