import React, { useState, useRef, useEffect } from 'react';
import StockChart from './StockChart';

/**
 * Expects `visualization.data` = an array of sequence dicts, e.g.:
 * [
 *   {
 *     id: 81009,
 *     sequence_set: 9,
 *     metadata: {
 *       ticker: "AAPL"
 *     },
 *     start_timestamp: "2022-10-20T00:00:00Z",
 *     end_timestamp: "2022-12-30T00:00:00Z",
 *     sequence_length: 50,
 *     sliced_data: [
 *       [open, high, low, close, volume],
 *       ...
 *     ]
 *   },
 *   ...
 * ]
 * and `visualization.config = { title, xAxisLabel, yAxisLabel }` (optional).
 *
 * Renders one StockChart at a time, with left/right arrows, plus
 * an input where the user can type a 1-based index and press Enter
 * to jump to that sequence. The code ensures Hooks are not called conditionally.
 */
const MultiStockChart = ({ visualization }) => {
  // 1) Declare all Hooks at the top, unconditionally.
  const [currentIndex, setCurrentIndex] = useState(0);
  const inputRef = useRef(null);

  // We'll do the effect here, unconditionally too
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.value = currentIndex + 1; // show 1-based index
    }
  }, [currentIndex]);

  // 2) Then read the data from props
  const sequenceDicts = visualization.data;
  const chartConfig = visualization.config || {};

  // 3) If there's no data, return early, but only after the hooks above have already run
  if (!Array.isArray(sequenceDicts) || sequenceDicts.length === 0) {
    return <div>No sequences to display</div>;
  }

  const totalCharts = sequenceDicts.length;

  // 4) Navigation with arrow buttons
  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, totalCharts - 1));
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  };

  // 5) Parse typed value on Enter
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (!inputRef.current) return;
      let newVal = parseInt(inputRef.current.value, 10);
      if (isNaN(newVal)) newVal = 1;
      // Clamp between 1 and totalCharts
      newVal = Math.max(1, Math.min(newVal, totalCharts));
      setCurrentIndex(newVal - 1); // convert to 0-based
    }
  };

  // 6) Current sequence dictionary
  const currentSequence = sequenceDicts[currentIndex];

  // Format timestamps as YYYY-MM-DD
  const formattedStart = currentSequence.start_timestamp
    ? new Date(currentSequence.start_timestamp).toISOString().split('T')[0]
    : '';
  const formattedEnd = currentSequence.end_timestamp
    ? new Date(currentSequence.end_timestamp).toISOString().split('T')[0]
    : '';

  // 7) Build a 'visualization' object for the StockChart
  const stock_visualization = {
    config: {
      title: chartConfig.title || `Stock Chart (Sequence #${currentIndex + 1})`,
      xAxisLabel: chartConfig.xAxisLabel || 'Index',
      yAxisLabel: chartConfig.yAxisLabel || 'Price',
    },
    data: currentSequence.sliced_data, // The [timeSteps, 5] array
  };

  // 8) Render
  return (
    <div style={{ color: '#fff' }}>
      {/* Row with left arrow, input "i of N" and right arrow */}
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

        {/* Middle: "Showing Sequence [inputRef] of N" */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: '0.5rem' }}>Showing Sequence</span>

          <input
            type="number"
            style={{ width: '3rem', textAlign: 'center' }}
            defaultValue={currentIndex + 1}  // initial display
            min="1"
            max={totalCharts}
            onKeyDown={handleKeyDown}
            ref={inputRef} // uncontrolled input reference
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

      {/* Display ticker/start/end on a single line (if present) */}
      {((currentSequence.metadata && currentSequence.metadata.ticker) ||
        formattedStart ||
        formattedEnd) && (
        <div style={{ marginBottom: '0.5rem', color: '#ccc' }}>
          {currentSequence.metadata?.ticker && (
            <span style={{ marginRight: '1rem', color: 'limegreen' }}>
              {currentSequence.metadata.ticker}
            </span>
          )}
          {formattedStart && (
            <span style={{ marginRight: '1rem' }}>
              <strong>Start:</strong> {formattedStart}
            </span>
          )}
          {formattedEnd && (
            <span>
              <strong>End:</strong> {formattedEnd}
            </span>
          )}
        </div>
      )}

      {/* Render the StockChart for the current sequence */}
      <StockChart visualization={stock_visualization} />
    </div>
  );
};

export default MultiStockChart;