import React, { useState, useRef, useEffect, Component } from 'react';
import StockChart from './StockChart';


/**
 * Expects visualization.data to be an object with the following keys:
 * {
 *   sequences: [ chartData1, chartData2, ... ],      // Each chartData is an array of [open, high, low, close, volume]
 *   predictions: [ predictions1, predictions2, ... ],  // Optional: each predictionsX is an array of numbers
 *   actual: [ actual1, actual2, ... ]                  // Optional: each actualX is an array of numbers
 * }
 * 
 * Also expects optionally: visualization.config = { title, xAxisLabel, yAxisLabel }
 * and optionally a metadata object at visualization.metadata.
 *
 * This component uses navigation controls to select the i-th sequence out of the arrays.
 */
const MultiStockChart = ({ visualization }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.value = currentIndex + 1; // display 1-based index
    }
  }, [currentIndex]);

  // Check that visualization.data exists, is an object, and contains a non-empty 'sequences' array.
  const dataObj = visualization.data;
  if (
    !dataObj ||
    typeof dataObj !== 'object' ||
    !Object.keys(dataObj).length === 0
  ) {
    return <div>No sequences to display</div>;
  }


  const totalCharts = dataObj[Object.keys(dataObj)[0]].length;

  // Navigation controls
  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, totalCharts - 1));
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  };

  // Allow the user to type in a 1-based sequence number and press Enter.
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (!inputRef.current) return;
      let newVal = parseInt(inputRef.current.value, 10);
      if (isNaN(newVal)) newVal = 1;
      newVal = Math.max(1, Math.min(newVal, totalCharts));
      setCurrentIndex(newVal - 1); // convert to 0-based index
    }
  };

  // Build a "currentSequence" object that mimics the expected structure.
  // For the primary chart data we pull the i-th element from the sequences array.
  const currentSequence = {
    // This must be an array that StockChart can use.
    sequence: dataObj[Object.keys(dataObj)[0]][currentIndex],
  };

  // If both 'predictions' and 'actual' exist and have an element at the current index, add them.
  if (
    dataObj.y_test &&
    dataObj.test_sequences &&
    Array.isArray(dataObj.pred_transformed) &&
    dataObj.y_test.length > currentIndex &&
    Array.isArray(dataObj.test_sequences) &&
    dataObj.y_test.length > currentIndex
  ) {
    currentSequence.predictions = dataObj.pred_transformed[currentIndex];
    currentSequence.actual = dataObj.y_test[currentIndex];
  }

  // Here we optionally use metadata from visualization.metadata.
  // (If not provided, the ticker, start, and end will not render.)

  // If any global timestamps exist at the data level (optional), format them.
  const formattedStart = currentSequence.sequence.start_timestamp
    ? new Date(currentSequence.sequence.start_timestamp).toISOString().split('T')[0]
    : '';
  const formattedEnd = currentSequence.sequence.end_timestamp
    ? new Date(currentSequence.sequence.end_timestamp).toISOString().split('T')[0]
    : '';
  
  const ticker = currentSequence.sequence.metadata.ticker;

  // Assemble the visualization object to pass to StockChart.
  const stock_visualization = {
    config: {
      title: visualization.config?.title || `Stock Chart (Sequence #${currentIndex + 1})`,
      xAxisLabel: visualization.config?.xAxisLabel || 'Index',
      yAxisLabel: visualization.config?.yAxisLabel || 'Price',
    },
    data: currentSequence.sequence.sliced_data,
  };

  // Instead of adding ticker and dates to the visualization.config,
  // pass them through metaData (using keys StockChart expects).
  const metaDataToPass = {
    ticker: ticker,
    startTimestamp: formattedStart,
    endTimestamp: formattedEnd,
  };

  // Pass along predictions and actual if they exist.
  if (
    currentSequence.hasOwnProperty('predictions') &&
    currentSequence.hasOwnProperty('actual')
  ) {
    stock_visualization.predictions = currentSequence.predictions;
    stock_visualization.actual = currentSequence.actual;
  }

  return (
    <div style={{ color: '#fff', display: 'flex', flexDirection: 'column', height: '100%', width: '100%' }}>
      {/* Navigation Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '0.5rem',
        }}
      >
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          style={{ padding: '0.2rem 0.5rem' }}
        >
          &larr;
        </button>

        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: '0.5rem' }}>Showing Sequence</span>
          <input
            type="number"
            style={{ width: '3rem', textAlign: 'center' }}
            defaultValue={currentIndex + 1}
            min="1"
            max={totalCharts}
            onKeyDown={handleKeyDown}
            ref={inputRef}
          />
          <span style={{ marginLeft: '0.5rem' }}>of {totalCharts}</span>
        </div>

        <button
          onClick={handleNext}
          disabled={currentIndex === totalCharts - 1}
          style={{ padding: '0.2rem 0.5rem' }}
        >
          &rarr;
        </button>
      </div>

      {/* Chart Container */}
      {/* This container will grow to fill all available space in the parent */}
      <div style={{ flex: 1, minHeight: 0 }}>
        {/* Pass the metaDataToPass object to StockChart */}
        <StockChart visualization={stock_visualization} metaData={metaDataToPass} />
      </div>
    </div>
  );
};

export default MultiStockChart;