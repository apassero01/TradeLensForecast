import React, { useState, useRef, useEffect } from 'react';

const StrategyList = ({ strategies, entityType, onSelect, onRefresh }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isListVisible, setIsListVisible] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const listRef = useRef(null);

  // Get entity-specific and generic strategies
  const entityTypeKey = entityType?.toLowerCase();
  const genericStrategies = strategies['entity'] || [];
  const entityStrategies = entityTypeKey ? (strategies[entityTypeKey] || []) : [];
  const allStrategies = [...genericStrategies, ...entityStrategies];

  // Filter strategies based on search term
  const filteredStrategies = allStrategies.filter(strategy => 
    strategy.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    setSearchTerm(strategy.name);
    setIsListVisible(false);
    onSelect(strategy);
  };

  const handleInputFocus = () => {
    setIsListVisible(true);
    setSelectedStrategy(null);
    setSearchTerm('');
    if (listRef.current) {
      listRef.current.scrollTop = 0; // Scroll to top when showing list
    }
  };

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
    setIsListVisible(true);
    setSelectedStrategy(null);
    if (listRef.current) {
      listRef.current.scrollTop = 0; // Scroll to top when searching
    }
  };

  // Make the handler more explicit and add a console.log
  const handleWheel = React.useCallback((e) => {
    e.stopPropagation();
    console.log("wheel event", e);
  }, []);

  return (
    <div className="flex flex-col">
      <div className="flex-none px-4 py-2 border-b border-gray-700">
        <div className="flex justify-between items-center mb-1">
          <h3 className="text-sm font-medium text-white">
            Select a Strategy
          </h3>
          <button
            onClick={onRefresh}
            className="text-gray-400 hover:text-white p-1"
            title="Refresh strategies"
          >
            ↻
          </button>
        </div>
        <div className="relative">
          <input
            type="text"
            placeholder="Search strategies..."
            value={searchTerm}
            onChange={handleSearchChange}
            onFocus={handleInputFocus}
            className="w-full bg-gray-700/50 border border-gray-600 rounded px-3 py-1
                     text-sm text-gray-200 placeholder-gray-400
                     focus:outline-none focus:border-gray-500"
          />
        </div>
      </div>

      {isListVisible && (
        <div 
          ref={listRef}
          className="overflow-y-auto h-[200px] scrollbar-none select-none"
        >
          <div className="space-y-1 p-2" onWheelCapture={handleWheel}>
            {filteredStrategies.map((strategy) => (
              <button
                key={strategy.name}
                className="w-full text-left px-2 py-1.5 bg-gray-700/30 hover:bg-gray-600/50 
                         text-gray-200 rounded transition-colors duration-150
                         border border-transparent hover:border-gray-600"
                onClick={() => handleStrategySelect(strategy)}
              >
                <div className="text-sm truncate">
                  {strategy.name}
                </div>
              </button>
            ))}
            
            {filteredStrategies.length === 0 && (
              <div className="text-sm text-gray-400 text-center py-4">
                {searchTerm ? 'No matching strategies found' : 'No strategies available'}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategyList; 