// src/components/Canvas/StrategyRequestEntity.jsx
import React, { memo } from 'react';
import EntityNodeBase from './EntityNodeBase';
import StrategyEditor from '../../Strategy/StrategyEditor';

function StrategyRequestEntity({ data }) {
  return (
    <EntityNodeBase 
      data={data}
    >
      {({ entity }) => (
        <div className="flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
          <div className="h-full w-full px-6 overflow-hidden">
            <StrategyEditor 
              existingRequest={{
                strategy_name: data.strategy_name,
                param_config: data.param_config,  
                target_entity_id: data.target_entity_id,
                add_to_history: data.add_to_history,
                nested_requests: data.nested_requests,
              }}
              entityType={entity.entity_type}
            />
          </div>
        </div>
      )}
    </EntityNodeBase>
  );
}

export default memo(StrategyRequestEntity);