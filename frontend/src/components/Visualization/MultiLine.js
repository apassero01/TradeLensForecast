import React from 'react';
import Line from './Line';

function MultiLine({ dataset, numRows = 2 }) {
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
      className="multi-line grid gap-4"
      style={{
        gridTemplateColumns: `repeat(${numColumns}, minmax(0, 1fr))`,
      }}
    >
      {features.map((feature, index) => (
        <div key={index}>
          <Line
            data={dataset[feature]}
            title={`Line Plot of ${feature}`}
          />
        </div>
      ))}
    </div>
  );
}

export default MultiLine;