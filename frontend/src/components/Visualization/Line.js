import React from 'react';
import ReactApexChart from 'react-apexcharts';

// Helper to convert a hex color to rgba with a given alpha
const hexToRgba = (hex, alpha = 1) => {
  const hexClean = hex.replace('#', '');
  const r = parseInt(hexClean.substring(0, 2), 16);
  const g = parseInt(hexClean.substring(2, 4), 16);
  const b = parseInt(hexClean.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const Line = ({ visualization }) => {
  if (!visualization || !visualization.data) {
    return <div>No visualization data available</div>;
  }

  const { data, config } = visualization;
  const { lines } = data; // lines: array of { label, values }
  const { title, xAxisLabel, yAxisLabel } = config;

  // 1. Figure out how many lines we have
  const lineCount = lines.length;

  // 2. Decide if we show line labels, legend, or tooltips
  const showLabels = lineCount <= 3;
  const showLegend = lineCount <= 3;
  // If more than 3 lines, disable tooltips to avoid huge hover popups
  const showTooltip = lineCount <= 3;

  // 3. Generate x-axis categories from the length of the second dimension
  //    Assuming each lineObj.values has the same length
  const seqLength = lineCount > 0 ? lines[0].values.length : 0;
  // e.g., [0, 1, 2, ..., seqLength - 1]
  const xAxisCategories = Array.from({ length: seqLength }, (_, i) => i);

  // 4. Define base colors
  const baseColors = [
    '#4CAF50', // Green
    '#FF9800', // Orange
    '#2196F3', // Blue
    '#FF5722', // Deep Orange
    '#9C27B0', // Purple
    '#00BCD4', // Cyan
  ];

  // If more than 3 lines, make them semi-transparent
  const colorSet = lineCount > 3
    ? baseColors.map((hex) => hexToRgba(hex, 0.4))
    : baseColors;

  // 5. Build the ApexCharts series with optional labeling & rounding
  const series = lines.map((lineObj) => {
    const label = showLabels ? lineObj.label : '';
    const roundedValues = lineObj.values.map((val) =>
      // Round each y-value to 2 decimal places
      parseFloat(val.toFixed(2))
    );
    return {
      name: label,
      data: roundedValues,
    };
  });

  // 6. Construct ApexCharts configuration
  const chartOptions = {
    chart: {
      type: 'line',
      height: 350,
      width: '100%',
      toolbar: { show: false },
      background: '#1e1e1e',
    },
    stroke: {
      curve: 'straight',
      width: 2,
    },
    colors: colorSet,
    xaxis: {
      categories: xAxisCategories,
      labels: {
        rotate: -45,
        style: { colors: '#9E9E9E' },
      },
      title: {
        text: xAxisLabel || 'X Axis',
        style: { color: '#9E9E9E' },
      },
    },
    yaxis: {
      title: {
        text: yAxisLabel || 'Value',
        style: { color: '#9E9E9E' },
      },
      labels: { style: { colors: '#9E9E9E' } },
    },
    grid: {
      borderColor: '#444',
    },
    tooltip: {
      enabled: showTooltip,   // Disable tooltips if > 3 lines
      shared: false,          // Don't aggregate all lines into one big popup
      intersect: true,        // Show only the hovered line's data
      theme: 'dark',
    },
    legend: {
      show: showLegend,
      position: 'top',
      labels: {
        colors: '#9E9E9E',
      },
    },
  };

  return (
    <div>
      {title && <h2 className="text-center text-gray-200 mb-1">{title}</h2>}
      <ReactApexChart
        options={chartOptions}
        series={series}
        type="line"
        height={350}
      />
    </div>
  );
};

export default Line;