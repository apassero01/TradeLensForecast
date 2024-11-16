import React from 'react';
import Histogram from './Histogram';

function MultiHistogram({ dataset, numRows = 2 }) {
  console.log('Dataset:', dataset);
  if (!dataset || typeof dataset !== 'object') {
    return <p className="text-red-500">Invalid dataset provided.</p>;
  }

  const features = Object.keys(dataset);
  const numFeatures = features.length;

  // Compute the number of columns
  const numColumns = Math.ceil(numFeatures / numRows);

  return (
    <div
      className="multi-histogram grid gap-4"
      style={{
        gridTemplateColumns: `repeat(${numColumns}, minmax(0, 1fr))`,
      }}
    >
      {features.map((feature, index) => (
        <div key={index}>
          <Histogram
            data={dataset[feature]}
            title={`Histogram of ${feature}`}
          />
        </div>
      ))}
    </div>
  );
}

export default MultiHistogram;