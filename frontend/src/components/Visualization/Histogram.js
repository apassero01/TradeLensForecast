import React from 'react';
import ReactApexChart from 'react-apexcharts';

const Histogram = ({ visualization }) => {
  if (!visualization || !visualization.data) {
    return <div>No visualization data available</div>;
  }

  const { data, config } = visualization;
  const { bins, counts } = data;
  const { title, xAxisLabel, yAxisLabel } = config;

  // Create bin labels from edges (using the left edge of each bin)
  const binLabels = bins.slice(0, -1).map(edge => edge.toFixed(2));

  const chartOptions = {
    chart: {
      type: 'bar',
      height: 350,
      width: '100%',
      toolbar: { show: false },
      background: '#1e1e1e',
    },
    plotOptions: {
      bar: {
        borderRadius: 0,
        columnWidth: '98%',
      },
    },
    colors: ['#4CAF50'],
    dataLabels: { enabled: false },
    series: [{ 
      name: 'Frequency', 
      data: counts 
    }],
    xaxis: {
      categories: binLabels,
      labels: { 
        rotate: -45, 
        style: { colors: '#9E9E9E' } 
      },
      title: { 
        text: xAxisLabel || 'Value', 
        style: { color: '#9E9E9E' } 
      },
    },
    yaxis: {
      title: { 
        text: yAxisLabel || 'Frequency', 
        style: { color: '#9E9E9E' } 
      },
      labels: { style: { colors: '#9E9E9E' } },
    },
    grid: { borderColor: '#444' },
    tooltip: { 
      theme: 'dark',
      y: {
        title: {
          formatter: () => 'Frequency:'
        }
      },
      x: {
        formatter: (val, opts) => {
          const binIndex = opts.dataPointIndex;
          return `Range: ${bins[binIndex].toFixed(2)} - ${bins[binIndex + 1].toFixed(2)}`;
        }
      }
    },
  };

  return (
    <div>
      <h2 className="text-center text-gray-200 mb-1">{title}</h2>
      <ReactApexChart 
        options={chartOptions} 
        series={chartOptions.series} 
        type="bar" 
        height={350} 
      />
    </div>
  );
};

export default Histogram;