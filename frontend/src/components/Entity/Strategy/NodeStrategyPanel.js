import React, { useState } from 'react';
import StrategyRequest from '../../../utils/StrategyRequest';

const StrategyIndicator = ({ request, index, total, onExecute, onEditRequest }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  // Calculate vertical position for stacking
  const spacing = 40; // Vertical space between indicators
  const startY = -((total - 1) * spacing) / 2; // Center the stack
  const top = startY + (index * spacing);

  const handleExecute = async (e) => {
    e.stopPropagation();
    console.log('StrategyIndicator executing with request:', request);
    const strategyRequest = new StrategyRequest({
      name: request.strategy_name,
      config: request.param_config,
      nested_requests: request.nested_requests,
      add_to_history: request.add_to_history,
      target_entity_id: request.target_entity_id,
      entity_id: request.entity_id
    });
    console.log('Created strategyRequest:', strategyRequest.toJSON());
    await onExecute(strategyRequest);
  };

  return (
    <div
      className="absolute transform pointer-events-auto"
      style={{
        top: `calc(50% + ${top}px)`,
        right: '-40px',
        zIndex: 20
      }}
    >
      <div className="relative">
        <div 
          className="relative cursor-pointer"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          {/* Expanding label */}
          {isHovered && (
            <div 
              className="absolute top-1/2 -translate-y-1/2 flex items-center bg-gray-900 
                         border border-gray-700 overflow-hidden whitespace-nowrap"
              style={{
                height: '32px',
                transform: 'translateY(-50%)',
                left: '28px', // Position to right of triangle
                zIndex: 1
              }}
            >
              <div className="px-3 text-xs text-white">
                {request.strategy_name}
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onEditRequest(request);
                }}
                className="px-3 h-full border-l border-gray-700 text-xs text-blue-400 
                           hover:text-blue-300 hover:bg-gray-800"
              >
                Edit
              </button>
            </div>
          )}
          
          {/* Triangle */}
          <div
            className="w-0 h-0 relative"
            style={{
              borderTop: '18px solid transparent',
              borderBottom: '18px solid transparent',
              borderLeft: '28px solid #22C55E',
              transition: 'border-left-color 0.2s',
              zIndex: 2
            }}
            onClick={handleExecute}
            onMouseEnter={(e) => e.currentTarget.style.borderLeftColor = '#16A34A'}
            onMouseLeave={(e) => e.currentTarget.style.borderLeftColor = '#22C55E'}
          />
        </div>
      </div>
    </div>
  );
};

const NodeStrategyPanel = ({ strategyRequests = [], onExecute, onEditRequest }) => {
  return (
    <div className="absolute inset-0">
      <div className="relative w-full h-full">
        {strategyRequests.map((request, index) => (
          <StrategyIndicator
            key={index}
            request={request}
            index={index}
            total={strategyRequests.length}
            onExecute={onExecute}
            onEditRequest={onEditRequest}
          />
        ))}
      </div>
    </div>
  );
};

export default NodeStrategyPanel; 