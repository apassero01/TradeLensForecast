import React, { useState, useRef, useEffect } from 'react';
import StockChart from './StockChart';

const MultiStockChart = ({ data, updateEntity, sendStrategyRequest }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const inputRef = useRef(null);
  // Enhanced drag tracking
  const [isDragging, setIsDragging] = useState(false);
  const dragStartX = useRef(null);
  const lastDragX = useRef(null);
  const dragDistance = useRef(0);
  const dragThreshold = 30; // Lower threshold for more responsive dragging

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.value = currentIndex + 1; // display 1-based index
    }
  }, [currentIndex]);

  const sequences = data?.sequences;
  const predictions = data?.predictions;
  const actuals = data?.actuals;

  // Check that visualization.data exists
  if (!sequences) {
    return <div>No sequences to display</div>;
  }

  const totalCharts = sequences.length;

  // Navigation controls
  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, totalCharts - 1));
  };

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  };

  // Allow the user to type in a sequence number
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (!inputRef.current) return;
      let newVal = parseInt(inputRef.current.value, 10);
      if (isNaN(newVal)) newVal = 1;
      newVal = Math.max(1, Math.min(newVal, totalCharts));
      setCurrentIndex(newVal - 1); // convert to 0-based index
    }
  };

  // Callback function for StockChart to use
  const handleChartDrag = (dragAmount) => {
    // Update drag distance and check for threshold crossing
    dragDistance.current += dragAmount;
    
    if (Math.abs(dragDistance.current) > dragThreshold) {
      if (dragDistance.current < 0 && currentIndex < totalCharts - 1) {
        // Dragged left, go to next chart
        handleNext();
        // Reset drag distance but keep dragging
        dragDistance.current = 0;
      } else if (dragDistance.current > 0 && currentIndex > 0) {
        // Dragged right, go to previous chart
        handlePrevious();
        // Reset drag distance but keep dragging
        dragDistance.current = 0;
      }
    }
  };

  // Reset drag tracking
  const resetDrag = () => {
    dragDistance.current = 0;
  };

  // Build visualization data structure
  const currentSequence = {
    sequence: sequences[currentIndex],
  };

  if (
    predictions &&
    actuals &&
    predictions.length > currentIndex &&
    actuals.length > currentIndex
  ) {
    currentSequence.predictions = predictions[currentIndex];
    currentSequence.actual = actuals[currentIndex];
  }

  // Format metadata
  const formattedStart = currentSequence.sequence.start_timestamp
    ? new Date(currentSequence.sequence.start_timestamp).toISOString().split('T')[0]
    : '';
  const formattedEnd = currentSequence.sequence.end_timestamp
    ? new Date(currentSequence.sequence.end_timestamp).toISOString().split('T')[0]
    : '';
  
  const ticker = currentSequence.sequence.metadata?.ticker;

  // Prepare stock visualization data
  const stock_visualization = {
    config: {
      title: `Stock Chart (Sequence #${currentIndex + 1})`,
      xAxisLabel: 'Index',
      yAxisLabel: 'Price',
    },
    data: currentSequence.sequence.sliced_data,
  };

  if (
    currentSequence.hasOwnProperty('predictions') &&
    currentSequence.hasOwnProperty('actual')
  ) {
    stock_visualization.predictions = currentSequence.predictions;
    stock_visualization.actual = currentSequence.actual;
  }

  // Pass chart navigation callbacks
  const metaDataToPass = {
    ticker: ticker,
    startTimestamp: formattedStart,
    endTimestamp: formattedEnd,
    dragHandlers: {
      onDrag: handleChartDrag,
      onDragEnd: resetDrag,
      canGoNext: currentIndex < totalCharts - 1,
      canGoPrevious: currentIndex > 0
    }
  };

  return (
    <div 
      className="nodrag" 
      style={{ 
        padding: '8px', 
        height: '100%', 
        width: '100%',
        userSelect: 'none'
      }}
    >
      {/* Metadata in top-left corner */}
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column',
        fontSize: '12px',
        color: '#ccc',
        marginBottom: '8px'
      }}>
        {ticker && <div><strong>Ticker:</strong> {ticker}</div>}
        {formattedStart && <div><strong>Start:</strong> {formattedStart}</div>}
        {formattedEnd && <div><strong>End:</strong> {formattedEnd}</div>}
      </div>

      {/* Navigation Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: '0.5rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <span style={{ marginRight: '0.5rem' }}>Showing Sequence</span>
          
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <button
              className="nodrag"
              onClick={(e) => {
                e.stopPropagation();
                handlePrevious();
              }}
              disabled={currentIndex === 0}
              style={{ 
                padding: '2px 6px',
                height: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '3px 0 0 3px',
                border: '1px solid #555',
                borderRight: 'none',
                background: '#333',
                cursor: currentIndex === 0 ? 'default' : 'pointer'
              }}
            >
              &larr;
            </button>
            
            <input
              className="nodrag"
              type="number"
              style={{ 
                width: '3rem', 
                textAlign: 'center',
                backgroundColor: '#333',
                color: 'white',
                border: '1px solid #555',
                padding: '2px 4px',
                fontSize: '14px',
                fontWeight: 'bold',
                borderRadius: '0',
                height: '24px'
              }}
              defaultValue={currentIndex + 1}
              min="1"
              max={totalCharts}
              onKeyDown={(e) => {
                e.stopPropagation();
                handleKeyDown(e);
              }}
              onClick={(e) => e.stopPropagation()}
              ref={inputRef}
            />
            
            <button
              className="nodrag"
              onClick={(e) => {
                e.stopPropagation();
                handleNext();
              }}
              disabled={currentIndex === totalCharts - 1}
              style={{ 
                padding: '2px 6px',
                height: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '0 3px 3px 0',
                border: '1px solid #555',
                borderLeft: 'none',
                background: '#333',
                cursor: currentIndex === totalCharts - 1 ? 'default' : 'pointer'
              }}
            >
              &rarr;
            </button>
          </div>
          
          <span style={{ marginLeft: '0.5rem' }}>of {totalCharts}</span>
        </div>
      </div>

      {/* Chart Container */}
      <div 
        className="nodrag nowheel" 
        style={{ 
          height: 'calc(100% - 100px)'
        }}
      >
        <StockChart 
          visualization={stock_visualization} 
          metaData={metaDataToPass}
        />
      </div>
    </div>
  );
};

export default MultiStockChart;