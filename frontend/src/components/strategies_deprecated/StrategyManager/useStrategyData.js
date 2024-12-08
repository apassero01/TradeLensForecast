// useStrategyData.js
import { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';

function useStrategyData({
  fetchAvailableEndpoint,
  submitEndpoint,
  sessionState,
  updateSessionState,
  strategyKey,
  options = {},
}) {
  const {
    manageTempStrategies = true,
    manageExistingStrategies = true,
    returnResponseData = false, // New option to control returning response data
  } = options;

  const [availableStrategies, setAvailableStrategies] = useState([]);
  const [tempStrategies, setTempStrategies] = useState([]);
  const [responseData, setResponseData] = useState(null); // State to hold response data
  const [error, setError] = useState(null);

  // Conditionally initialize existingStrategies
  const existingStrategies = manageExistingStrategies
    ? (sessionState.sessionData?.[strategyKey] || []).map((strategy) => ({
        ...strategy,
        id: strategy.id || uuidv4(),
      }))
    : [];

  // Fetch available strategies on mount
  useEffect(() => {
    const fetchAvailableStrategies = async () => {
      try {
        const response = await fetch(fetchAvailableEndpoint);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const strategies = data.map((strategy) => ({ ...strategy, id: uuidv4() }));
        setAvailableStrategies(strategies);
      } catch (err) {
        setError('Failed to load available strategies');
        console.error(err);
      }
    };
    fetchAvailableStrategies();
  }, [fetchAvailableEndpoint]);

  // Handler to add a strategy to tempStrategies
  const handleAddStrategy = (strategy) => {
    if (!manageTempStrategies) return;
    const strategyWithId = { ...strategy, id: uuidv4() };
    setTempStrategies((prevStrategies) => [...prevStrategies, strategyWithId]);
  };

  // Handler to remove a strategy from tempStrategies
  const handleRemoveTempStrategy = (id) => {
    if (!manageTempStrategies) return;
    setTempStrategies((prevStrategies) => prevStrategies.filter((s) => s.id !== id));
  };

  // Default handler to submit a strategy to the backend
  const handleSubmit = async (strategy) => {
    try {
      const response = await fetch(submitEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(strategy),
      });

      if (response.ok) {
        const data = await response.json();
        const updatedSessionData = data.sessionData;

        if (updatedSessionData) {
          updateSessionState('sessionData', updatedSessionData);
          if (manageTempStrategies) {
            handleRemoveTempStrategy(strategy.id);
          }
          if (returnResponseData) {

            setResponseData({ strategy, data }); // Store response data in state
          }
          return data; // Return response data
        } else {
          alert('Failed to update session data');
        }
      } else {
        const errorData = await response.json();
        alert('Failed to submit: ' + errorData.error);
      }
    } catch (err) {
      console.error(err);
      alert('Failed to submit');
    }
  };

  return {
    availableStrategies,
    tempStrategies: manageTempStrategies ? tempStrategies : [],
    existingStrategies,
    responseData, // Expose response data
    error,
    handleAddStrategy: manageTempStrategies ? handleAddStrategy : undefined,
    handleRemoveTempStrategy: manageTempStrategies ? handleRemoveTempStrategy : undefined,
    handleSubmit,
  };
}

export default useStrategyData;