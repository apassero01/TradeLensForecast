import React, { useState, useEffect } from 'react';
import ReactApexChart from 'react-apexcharts';

const Histogram = ({ data, title, numBins = 25 }) => {
  const [chartData, setChartData] = useState({ bins: [], labels: [] });
  const [stats, setStats] = useState({ mean: 0, variance: 0 });

  useEffect(() => {
    if (data && data.length > 0) {
      const min = Math.min(...data);
      const max = Math.max(...data);
      const binWidth = (max - min) / numBins;

      const bins = new Array(numBins).fill(0);
      const labels = Array.from({ length: numBins }, (_, i) => {
        const start = (min + i * binWidth).toFixed(2);
        return `${start}`;
      });

      data.forEach(value => {
        const binIndex = Math.min(Math.floor((value - min) / binWidth), numBins - 1);
        bins[binIndex]++;
      });

      // Set histogram data
      setChartData({ bins, labels });

      // Calculate mean
      const mean = data.reduce((acc, val) => acc + val, 0) / data.length;

      // Calculate variance
      const variance =
        data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;

      // Update stats
      setStats({ mean, variance });
    }
  }, [data, numBins]);

  const maxFrequency = Math.max(...chartData.bins) || 0;
  const yMax = Math.ceil(maxFrequency * 1.1);

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
    series: [{ name: 'Frequency', data: chartData.bins }],
    xaxis: {
      categories: chartData.labels,
      labels: { rotate: -45, style: { colors: '#9E9E9E' } },
      title: { text: 'Value', style: { color: '#9E9E9E' } },
    },
    yaxis: {
      max: yMax,
      title: { text: 'Frequency', style: { color: '#9E9E9E' } },
      labels: { style: { colors: '#9E9E9E' } },
    },
    grid: { borderColor: '#444' },
    tooltip: { theme: 'dark' },
  };

  return (
    <div>
      <h2 className="text-center text-gray-200 mb-1">{title}</h2>
      <p className="text-center text-gray-400 mb-4">
        Mean: {stats.mean.toFixed(2)} | Variance: {stats.variance.toFixed(2)}
      </p>
      <ReactApexChart options={chartOptions} series={chartOptions.series} type="bar" height={350} />
    </div>
  );
};

export default Histogram;