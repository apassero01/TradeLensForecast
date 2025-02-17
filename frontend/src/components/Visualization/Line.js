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
    // Decide whether or not to show the label based on the number of lines
    const label = showLabels ? lineObj.label : '';

    // 1. Convert each value to its absolute value
    const absoluteValues = lineObj.values.map((val) => Math.abs(val));

    // 2. Compute the mean for the current time series using absolute values
    const mean =
      absoluteValues.reduce((sum, val) => sum + val, 0) / absoluteValues.length;

    // 3. Define thresholds:
    //    - magnitudeThreshold: if the absolute difference between the mean and a value is at least 10% of the mean, treat it as an outlier.
    //    - largeMagnitudeCutoff: perform replacement only if the mean is above this value.
    const magnitudeThreshold = mean * 0.1; // Adjust this factor as needed
    const largeMagnitudeCutoff = 100;       // Adjust this cutoff based on your data's scale

    // 4. For each value:
    //    - Round it to 2 decimals.
    //    - If the mean is greater than our large magnitude cutoff and the absolute difference from the mean is at least the threshold,
    //      then replace the value with the (rounded) mean.
    const adjustedValues = absoluteValues.map((val) => {
      let rounded = parseFloat(val.toFixed(2));
      if (mean > largeMagnitudeCutoff && Math.abs(mean - rounded) >= magnitudeThreshold) {
        rounded = parseFloat(mean.toFixed(2));
      }
      return rounded;
    });

    return {
      name: label,
      data: adjustedValues,
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
      tickAmount: 10, // Limits the number of tick marks on the x-axis
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