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

const NewLine = ({ data, sendStrategyRequest, updateEntity, viewEntityId, parentEntityId }) => {
  // data prop is expected to contain: { array, title?, xAxisLabel?, yAxisLabel? }
  if (!data || !data.array || data.array.length === 0) {
    return <div>No data array available for chart.</div>;
  }

  const { array, title, xAxisLabel, yAxisLabel } = data;

  // Assuming the input 'array' is a simple array of numbers for a single line
  // If 'array' can contain multiple lines, this logic will need adjustment
  const seriesData = array.map(val => parseFloat(val.toFixed(2)));
  const seqLength = seriesData.length;
  const xAxisCategories = Array.from({ length: seqLength }, (_, i) => i);

  // Define base colors - using a single color for a single line
  const baseColor = '#4CAF50'; // Green

  const series = [
    {
      name: title || 'Data', // Use title for series name if available
      data: seriesData,
    },
  ];

  const chartOptions = {
    chart: {
      type: 'line',
      height: 350,
      width: '100%',
      toolbar: { show: false },
      background: '#1e1e1e', // Matching dark theme
    },
    stroke: {
      curve: 'straight',
      width: 2,
    },
    colors: [baseColor],
    xaxis: {
      categories: xAxisCategories,
      tickAmount: 10,
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
      enabled: true,
      shared: false,
      intersect: true,
      theme: 'dark',
    },
    legend: {
      show: false, // Assuming single line, so legend might not be needed
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

export default NewLine; 