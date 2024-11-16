import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';

const Line = ({ data, title }) => {
  const [chartData, setChartData] = useState([]);
  useEffect(() => {
    if (data && data.length > 0 && data[0].length > 0) {
      // Prepare data for each line in the 2D array
      const seriesData = data.map((lineData, index) => ({
        name: `Line ${index + 1}`,
        data: lineData.map(value => parseFloat(value.toFixed(2))), // Round each value to 2 decimal places
      }));

      setChartData(seriesData);
    }
  }, [data]);

  const chartOptions = {
    chart: {
      type: 'line',
      height: 350,
      width: '100%',
      toolbar: { show: false },
      background: '#1e1e1e',
    },
    stroke: {
      curve: 'smooth',
      width: 2,
    },
    colors: ['#FF5733', '#33B5FF', '#FF33FF', '#FFC733', '#33FF57'], // Adjust colors for multiple lines
    dataLabels: { enabled: false },
    xaxis: {
      categories: Array.from({ length: data[0].length }, (_, i) => i + 1), // Time steps on the x-axis
      title: { text: 'Time Steps', style: { color: '#9E9E9E' } },
      labels: { style: { colors: '#9E9E9E' } },
    },
    yaxis: {
      title: { text: 'Value', style: { color: '#9E9E9E' } },
      labels: {
        style: { colors: '#9E9E9E' },
        formatter: value => value.toFixed(2), // Round y-axis labels to 2 decimal places
      },
    },
    grid: { borderColor: '#444' },
    tooltip: {
      theme: 'dark',
      y: {
        formatter: value => value.toFixed(2), // Round tooltip values to 2 decimal places
      },
    },
  };



  return (
    <div>
      <h2 className="text-center text-gray-200 mb-1">{title}</h2>
      <ReactApexChart options={chartOptions} series={chartData} type="line" height={350} />
    </div>
  );
};

export default Line;