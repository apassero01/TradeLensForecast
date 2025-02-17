import React, { useState } from 'react';
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
    const predictionsData = visualization.predictions.map((value, i) => ({
      x: baseLength + i,
      y: parseFloat(value[0].toFixed(2)),
    }));
    extraSeries.push({
      name: 'Predictions',
      type: 'line',
      data: predictionsData,
    });
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
    tooltip: { theme: 'dark' },
    plotOptions: {
      candlestick: { wick: { useFillColor: true } },
    },
    legend: {
      position: 'top',
      labels: { colors: '#9E9E9E' },
    },
  };

  return (
    <div style={{ margin: '1rem 0', flex: 1, height: '100%', width: '100%' }}>
      {/* Render the meta-information */}
      <MetaInformation ticker={ticker} startTimestamp={startTimestamp} endTimestamp={endTimestamp} />

      {/* Control Panel to toggle extra series */}
      <div style={{ marginBottom: '1rem', color: '#fff' }}>
        <label style={{ marginRight: '1rem' }}>
          <input
            type="checkbox"
            checked={showPredictions}
            onChange={(e) => setShowPredictions(e.target.checked)}
            style={{ marginRight: '0.3rem' }}
          />
          Show Predictions
        </label>
        <label>
          <input
            type="checkbox"
            checked={showActual}
            onChange={(e) => setShowActual(e.target.checked)}
            style={{ marginRight: '0.3rem' }}
          />
          Show Actual
        </label>
      </div>

      {/* Wrap the chart in a container that applies padding around the chart */}
      <div style={{ padding: '8px', height: '100%', width: '100%', boxSizing: 'border-box' }}>
        <ReactApexChart
          options={chartOptions}
          series={finalSeries}
          type={chartType}
          height="100%"
          width="100%"
        />
      </div>
    </div>
  );
};

export default StockChart;