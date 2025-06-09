import React, { useState, useRef, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';

/**
 * MetaInformation Component
 *
 * Displays ticker and date range if provided.
 */
const MetaInformation = ({ ticker, startTimestamp, endTimestamp }) => {
  if (!ticker && !startTimestamp && !endTimestamp) return null;
  return (
    <div style={{ marginBottom: '0.5rem', color: '#ccc' }}>
      {ticker && (
        <span style={{ marginRight: '1rem' }}>
          <strong>Ticker:</strong> {ticker}
        </span>
      )}
      {startTimestamp && (
        <span style={{ marginRight: '1rem' }}>
          <strong>Start:</strong> {startTimestamp}
        </span>
      )}
      {endTimestamp && (
        <span>
          <strong>End:</strong> {endTimestamp}
        </span>
      )}
    </div>
  );
};

/**
 * StockChart Component
 *
 * Renders a candlestick chart (or mixed chart) along with optional meta data.
 * Now includes checkboxes to toggle rendering of predictions and actual.
 */
const StockChart = ({ visualization, metaData = {} }) => {
  const [showPredictions, setShowPredictions] = useState(true);
  const [showActual, setShowActual] = useState(true);
  const chartContainerRef = useRef(null);
  
  // Extract drag handlers from metaData
  const dragHandlers = metaData.dragHandlers || {};

  // Add event handlers to stop propagation
  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container) return;

    // Function to stop event propagation
    const stopPropagation = (e) => {
      e.stopPropagation();
    };

    // Attach listeners for all relevant events that might trigger React Flow dragging
    const events = [
      'mousedown', 'mousemove', 'mouseup', 
      'touchstart', 'touchmove', 'touchend',
      'pointerdown', 'pointermove', 'pointerup'
    ];
    
    events.forEach(event => {
      container.addEventListener(event, stopPropagation, { capture: true });
    });

    // Clean up
    return () => {
      events.forEach(event => {
        container.removeEventListener(event, stopPropagation, { capture: true });
      });
    };
  }, []);

  console.log('visualization', visualization);
  if (!visualization || !visualization.data) {
    return <div>No stock data available</div>;
  }

  const { data, config = {} } = visualization;
  if (!Array.isArray(data) || data.length === 0) {
    return <div>No stock data available</div>;
  }

  const {
    title = 'Stock Chart',
    xAxisLabel = 'Time',
    yAxisLabel = 'Price',
  } = config;

  // Destructure metaData fields if provided.
  const { ticker, startTimestamp, endTimestamp } = metaData;
  const baseLength = data.length;

  // Build candlestick series data.
  const candlestickData = data.map((row, i) => {
    const [open, high, low, close] = row;
    return {
      x: i,
      y: [
        parseFloat(open.toFixed(2)),
        parseFloat(high.toFixed(2)),
        parseFloat(low.toFixed(2)),
        parseFloat(close.toFixed(2)),
      ],
    };
  });

  // Build extra series using the checkboxes to decide if we should render them.
  let extraSeries = [];
  if (showPredictions && visualization.predictions && Array.isArray(visualization.predictions)) {
    // Check if predictions have multiple dimensions
    const firstPrediction = visualization.predictions[0];
    const isMultiDimensional = Array.isArray(firstPrediction) && firstPrediction.length > 1;
    
    if (isMultiDimensional) {
      // Handle multi-dimensional predictions - create a series for each dimension
      const dimensions = firstPrediction.length;
      
      // Generate distinct colors for each dimension
      const predictionColors = [
        'rgba(0, 176, 255, 0.5)',    // Light blue with opacity
        'rgba(255, 173, 0, 0.5)',    // Orange with opacity
        'rgba(124, 207, 0, 0.5)',    // Green with opacity
        'rgba(255, 102, 255, 0.5)',  // Pink with opacity
        'rgba(255, 236, 0, 0.5)',    // Yellow with opacity
      ];
      
      for (let dim = 0; dim < dimensions; dim++) {
        const dimensionData = visualization.predictions.map((value, i) => ({
          x: baseLength + i,
          y: parseFloat(value[dim].toFixed(2)),
        }));
        
        extraSeries.push({
          name: `Prediction ${dim + 1}`,
          type: 'line',
          data: dimensionData,
          opacity: 0.6, // Reduced opacity for predictions
          color: predictionColors[dim % predictionColors.length], // Cycle through colors
          dashArray: 4, // Add dashed line style for predictions
        });
      }
    } else {
      // Original single-dimension handling
      const predictionsData = visualization.predictions.map((value, i) => ({
        x: baseLength + i,
        y: parseFloat(value[0].toFixed(2)),
      }));
      extraSeries.push({
        name: 'Predictions',
        type: 'line',
        data: predictionsData,
        opacity: 0.6, // Reduced opacity for predictions
        color: 'rgba(0, 176, 255, 0.5)', // Light blue with opacity
        dashArray: 4, // Add dashed line style for predictions
      });
    }
  }
  if (showActual && visualization.actual && Array.isArray(visualization.actual)) {
    const actualData = visualization.actual.map((value, i) => ({
      x: baseLength + i,
      y: parseFloat(value[0].toFixed(2)),
    }));
    extraSeries.push({
      name: 'Actual',
      type: 'line',
      data: actualData,
      color: '#00ff00', // Bright green for actual values
      lineWidth: 2, // Slightly thicker line for actual values
    });
  }

  // Determine if we have any extra series to render.
  const mixedChart = extraSeries.length > 0;
  // When mixed, we use 'line' as it supports multiple series types.
  const chartType = mixedChart ? 'line' : 'candlestick';

  // Compute the maximum extra range based on which extra series we are displaying.
  let extraRange = 0;
  if (showPredictions && visualization.predictions) {
    extraRange = Math.max(extraRange, visualization.predictions.length);
  }
  if (showActual && visualization.actual) {
    extraRange = Math.max(extraRange, visualization.actual.length);
  }

  // Final series array includes the OHLC series plus any extra series.
  const finalSeries = [
    {
      name: 'OHLC',
      type: 'candlestick',
      data: candlestickData,
    },
    ...extraSeries,
  ];

  // Configure the x-axis options.
  const xaxisOptions = {
    labels: { style: { colors: '#9E9E9E' } },
    title: { text: xAxisLabel, style: { color: '#9E9E9E' } },
    ...(extraRange > 0 && { min: 0, max: baseLength + extraRange - 1 }),
  };

  // Chart options with interactive zoom/pan and a subtitle.
  const chartOptions = {
    chart: {
      id: 'stockChart',
      type: chartType,
      background: '#1e1e1e',
      toolbar: {
        show: true,
        tools: {},
        autoSelected: 'zoom',
      },
      zoom: {
        enabled: true,
        type: 'x',
        autoScaleYaxis: true,
      },
      events: {
        // Disable ApexCharts built-in selection by stopping event propagation
        mouseDown: function(event) {
          event.stopPropagation();
        },
        mouseMove: function(event) {
          event.stopPropagation();
        },
        mouseUp: function(event) {
          event.stopPropagation();
        },
        touchStart: function(event) {
          event.stopPropagation();
        },
        touchMove: function(event) {
          event.stopPropagation(); 
        },
        touchEnd: function(event) {
          event.stopPropagation();
        }
      }
    },
    title: {
      text: title,
      align: 'left',
      style: { color: '#fff' },
    },
    subtitle: {
      text: `${ticker ? `Ticker: ${ticker}` : ''}${
        ticker && startTimestamp && endTimestamp ? ' | ' : ''
      }${startTimestamp && endTimestamp ? `Dates: ${startTimestamp} - ${endTimestamp}` : ''}`,
      align: 'left',
      style: { color: '#ccc', fontSize: '12px' },
    },
    xaxis: xaxisOptions,
    yaxis: {
      labels: { style: { colors: '#9E9E9E', fontSize: '12px' } },
      title: { text: yAxisLabel, style: { color: '#9E9E9E', fontSize: '24px' } },
    },
    grid: { borderColor: '#444' },
    plotOptions: {
      candlestick: { wick: { useFillColor: true } },
    },
    legend: {
      show: false,
      position: 'top',
      labels: { colors: '#9E9E9E' },
    },
    selection: {
      enabled: true,
      type: 'x',
      fill: {
        color: '#24292e',
        opacity: 0.1
      },
      stroke: {
        width: 1,
        dashArray: 3,
        color: '#24292e',
        opacity: 0.4
      },
    },
  };

  return (
    <div style={{ margin: '1rem 0', flex: 1, height: '100%', width: '100%' }}>
      {/* Render the meta-information */}
      <MetaInformation ticker={ticker} startTimestamp={startTimestamp} endTimestamp={endTimestamp} />

      {/* Control Panel to toggle extra series */}
      <div style={{ marginBottom: '1rem', color: '#fff' }}>
        <label style={{ marginRight: '1rem' }} className="nodrag">
          <input
            type="checkbox"
            className="nodrag"
            checked={showPredictions}
            onChange={(e) => {
              e.stopPropagation();
              setShowPredictions(e.target.checked);
            }}
            style={{ marginRight: '0.3rem' }}
          />
          Show Predictions
        </label>
        <label className="nodrag">
          <input
            type="checkbox"
            className="nodrag"
            checked={showActual}
            onChange={(e) => {
              e.stopPropagation();
              setShowActual(e.target.checked);
            }}
            style={{ marginRight: '0.3rem' }}
          />
          Show Actual
        </label>
      </div>

      {/* Wrap the chart in a container that applies padding around the chart and prevents drag */}
      <div 
        ref={chartContainerRef}
        className="nodrag nowheel" 
        style={{ 
          padding: '8px', 
          height: '100%', 
          width: '100%', 
          boxSizing: 'border-box',
          pointerEvents: 'auto',
          touchAction: 'none'
        }}
        onMouseDown={(e) => e.stopPropagation()}
        onTouchStart={(e) => e.stopPropagation()}
      >
        <ReactApexChart
          options={chartOptions}
          series={finalSeries}
          type={chartType}
          height="100%"
          width="100%"
          dragHandlers={dragHandlers}
        />
      </div>
    </div>
  );
};

export default StockChart;