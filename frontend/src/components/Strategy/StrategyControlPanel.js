import React, { useState, useRef, useEffect } from 'react';
import StrategyConfigEditor from './StrategyConfigEditor';

const StrategyControlPanel = ({ selectedEntity, availableStrategies, onExecuteStrategy }) => {
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [width, setWidth] = useState(520);
  const panelRef = useRef(null);
  const isDragging = useRef(false);
  const startX = useRef(0);
  const startWidth = useRef(0);

  // Handle mouse events for resizing
  const handleMouseDown = (e) => {
    e.preventDefault();
    isDragging.current = true;
    startX.current = e.pageX;
    startWidth.current = width;
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (e) => {
    if (!isDragging.current) return;
    const delta = startX.current - e.pageX;
    const newWidth = Math.max(400, Math.min(800, startWidth.current + delta));
    setWidth(newWidth);
  };

  const handleMouseUp = () => {
    isDragging.current = false;
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  // Cleanup event listeners
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);
  
  if (!selectedEntity) {
    return (
      <div 
        ref={panelRef}
        style={{ width: `${width}px` }}
        className="bg-gray-800 flex flex-col h-screen border-l border-gray-700 relative"
      >
        <div className="px-8 py-6">
          <h3 className="text-xl text-white font-semibold">
            Select an Entity
          </h3>
          <p className="text-sm text-gray-400 mt-1">
            Click on an entity in the graph to view available strategies
          </p>
        </div>
        <div 
          className="absolute left-0 top-0 h-full w-1 cursor-ew-resize hover:bg-blue-500/50"
          onMouseDown={handleMouseDown}
        />
      </div>
    );
  }
  
  // Get entity-specific and generic strategies
  const entityType = selectedEntity.data.label.toLowerCase();
  const genericStrategies = availableStrategies['entity'] || [];
  const entityStrategies = availableStrategies[entityType] || [];
  const strategies = [...genericStrategies, ...entityStrategies];

  console.log('Available strategies for', entityType, ':', strategies);

  return (
    <div 
      ref={panelRef}
      style={{ width: `${width}px` }}
      className="bg-gray-800 flex flex-col h-screen border-l border-gray-700 relative"
    >
      <div 
        className="absolute left-0 top-0 h-full w-1 cursor-ew-resize hover:bg-blue-500/50"
        onMouseDown={handleMouseDown}
      />
      
      {!selectedStrategy ? (
        <div className="flex flex-col h-full">
          <div className="px-8 py-6 border-b border-gray-700">
            <h3 className="text-xl text-white font-semibold">
              {selectedEntity.data.label}
            </h3>
            <p className="text-sm text-gray-400 mt-1">
              Select a strategy to configure
            </p>
          </div>

          <div className="flex-grow overflow-y-auto">
            <div className="space-y-3 px-8 py-6">
              {strategies.length > 0 ? (
                strategies.map((strategy) => (
                  <button
                    key={strategy.name}
                    className="w-full text-left px-6 py-4 bg-gray-700/50 hover:bg-gray-600/50 
                             text-white rounded-lg transition-colors duration-150
                             border border-gray-600 hover:border-gray-500"
                    onClick={() => setSelectedStrategy(strategy)}
                  >
                    <div className="font-medium">{strategy.name}</div>
                    <div className="text-sm text-gray-400 mt-1">
                      {strategy.config.strategy_name || 'Configure and execute this strategy'}
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-gray-400 text-center py-4">
                  No strategies available for this entity type
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col h-full">
          <div className="px-8 py-6 border-b border-gray-700">
            <button
              onClick={() => setSelectedStrategy(null)}
              className="text-gray-400 hover:text-white flex items-center gap-2
                       transition-colors duration-150"
            >
              <span>←</span>
              <span>Back to strategies</span>
            </button>
          </div>
          
          <div className="flex-grow overflow-y-auto">
            <div className="px-8 py-6">
              <StrategyConfigEditor
                strategy={selectedStrategy}
                entityType={selectedEntity.data.label}
                selectedEntity={selectedEntity}
                onSubmit={(config) => {
                  onExecuteStrategy(selectedEntity, selectedStrategy.name, config);
                  setSelectedStrategy(null);
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyControlPanel;
