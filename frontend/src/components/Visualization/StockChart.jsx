import React from 'react';
import ReactApexChart from 'react-apexcharts';

/**
 * StockChart Props:
 *  - visualization: {
 *      config: { title, xAxisLabel, yAxisLabel }, // optional chart labels
 *      data: [[open, high, low, close, volume], ...] // the candlestick data
 *    }
 *  - metaData: {
 *      ticker?: string,
 *      startTimestamp?: string,
 *      endTimestamp?: string
 *    }
 *
 * Renders a candlestick chart, rounding each price to 2 decimals.
 */
const StockChart = ({ visualization, metaData = {} }) => {
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

  // Destructure metaData fields
  const {
    ticker,
    startTimestamp,
    endTimestamp,
  } = metaData;

  // Build candlestick data (round each price to 2 decimals)
  const candlestickData = data.map((row, i) => {
    const [open, high, low, close] = row; // ignoring volume for chart geometry
    return {
      x: i, // or real date/time
      y: [
        parseFloat(open.toFixed(2)),
        parseFloat(high.toFixed(2)),
        parseFloat(low.toFixed(2)),
        parseFloat(close.toFixed(2)),
      ],
    };
  });

  const series = [
    {
      name: 'OHLC',
      data: candlestickData,
    },
  ];

  const chartOptions = {
    chart: {
      type: 'candlestick',
      height: 350,
      background: '#1e1e1e',
      toolbar: { show: true },
    },
    title: {
      text: title,
      align: 'left',
      style: { color: '#fff' },
    },
    xaxis: {
      labels: {
        style: { colors: '#9E9E9E' },
      },
      title: {
        text: xAxisLabel,
        style: { color: '#9E9E9E' },
      },
    },
    yaxis: {
      labels: {
        style: { colors: '#9E9E9E' },
      },
      title: {
        text: yAxisLabel,
        style: { color: '#9E9E9E' },
      },
    },
    grid: {
      borderColor: '#444',
    },
    tooltip: {
      theme: 'dark',
    },
    plotOptions: {
      candlestick: {
        wick: {
          useFillColor: true,
        },
      },
    },
    legend: {
      position: 'top',
      labels: { colors: '#9E9E9E' },
    },
  };

  return (
    <div style={{ margin: '1rem 0' }}>
      {/* Display meta info (ticker, start/end) above the chart */}
      {(ticker || startTimestamp || endTimestamp) && (
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
      )}

      <ReactApexChart
        options={chartOptions}
        series={series}
        type="candlestick"
        height={350}
      />
    </div>
  );
};

export default StockChart;