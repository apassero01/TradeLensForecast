import React, { useState, useEffect } from 'react';
import StrategyList from './StrategyList';
import StrategyHistory from './StrategyHistory';
import StrategyEditor from '../StrategyEditor';
import { strategyApi } from '../../../services/strategyApi';

const StrategyPanel = ({ 
  selectedEntity, 
  availableStrategies,
  onStrategyExecute,
  fetchAvailableStrategies,
  onStrategyListExecute,
}) => {
  const [historyItems, setHistoryItems] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const { strategy_requests } = await strategyApi.getHistory();
      setHistoryItems(strategy_requests);
    } catch (error) {
      console.error('Failed to fetch strategy history:', error);
    }
  };

  const handleHistorySelect = (historyItem) => {
    setSelectedHistoryItem(historyItem);
    setSelectedStrategy(null);
  };

  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    setSelectedHistoryItem(null);
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  const handleExecuteWithHistory = async (request) => {
    const response = await onStrategyExecute(request);  // Get response
    await fetchHistory();  // Update history
    return response;  // Pass response back through the chain
  };

  const handleExecuteListWithHistory = async (strategyList) => {
    console.log('Executing strategy list with history:', strategyList);
    const response = await onStrategyListExecute(strategyList);  // Get response
    await fetchHistory();  // Update history
    return response;  // Pass response back through the chain
  }

  return (
    <div 
      className="flex h-screen bg-gray-800 border-l border-gray-700 relative transition-all duration-300"
      style={{ width: isCollapsed ? '48px' : '800px' }}
    >
      {/* Collapse Toggle Button */}
      <button
        onClick={toggleCollapse}
        className="absolute left-2 top-4 text-gray-400 hover:text-white 
                 transition-colors duration-150 z-10"
      >
        {isCollapsed ? '→' : '←'}
      </button>

      {!isCollapsed && (
        <>
          {/* History Sidebar */}
          <div className="w-64 flex-shrink-0 border-r border-gray-700">
            <StrategyHistory
              items={historyItems}
              selectedItem={selectedHistoryItem}
              onSelect={handleHistorySelect}
              onListExecute={handleExecuteListWithHistory}
            />
          </div>

          {/* Main Content Area */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {!selectedEntity ? (
              <div className="p-6 text-gray-400">
                Select an entity to view strategies
              </div>
            ) : !selectedStrategy && !selectedHistoryItem ? (
              <div className="flex-1 overflow-y-auto">
                <StrategyList
                  strategies={availableStrategies}
                  entityType={selectedEntity.data.label}
                  onSelect={handleStrategySelect}
                  onRefresh={fetchAvailableStrategies}
                />
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto">
                <StrategyEditor
                  strategy={selectedStrategy}
                  historyItem={selectedHistoryItem}
                  selectedEntity={selectedEntity}
                  onExecute={handleExecuteWithHistory}
                  onBack={() => {
                    setSelectedStrategy(null);
                    setSelectedHistoryItem(null);
                  }}
                />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default StrategyPanel; 