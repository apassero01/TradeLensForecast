import React, { useState } from 'react';
import Line from './Line';

function SequenceMultiLine({ dataset, numRows = 2 }) {
  // Initialize state for current index
  const [currentIndex, setCurrentIndex] = useState(0);

  // Validate dataset
  if (!dataset || !Array.isArray(dataset) || dataset.length === 0) {
    return <p className="text-red-500">Invalid dataset provided.</p>;
  }

  const totalSequences = dataset.length;
  const currentItem = dataset[currentIndex];

  // Extract feature names from the current item
  const featureNames = Object.keys(currentItem).filter(
    (key) => key !== 'metadata'
  );

  // Ensure there are features to display
  if (featureNames.length === 0) {
    return <p className="text-red-500">No features to display.</p>;
  }

  // Extract metadata from the first feature
  const firstFeatureName = featureNames[0];
  const metadata = currentItem[firstFeatureName]?.metadata || {};
  const ticker = metadata?.metadata?.ticker || 'Unknown Ticker';
  const startTimestamp = metadata?.start_timestamp || 'Unknown Start Time';
  const endTimestamp = metadata?.end_timestamp || 'Unknown End Time';

  // Prepare data for each feature
  const featureDataMap = {}; // { featureName: dataArray }

  featureNames.forEach((featureName) => {
    const dataEntry = currentItem[featureName];
    if (dataEntry && dataEntry.data) {
      featureDataMap[featureName] = dataEntry.data;
    }
  });

  const features = Object.keys(featureDataMap);
  const numFeatures = features.length;

  // Compute the number of columns
  const numColumns = Math.ceil(numFeatures / numRows);

  // Navigation handlers
  const handlePrevious = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex > 0 ? prevIndex - 1 : totalSequences - 1
    );
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex < totalSequences - 1 ? prevIndex + 1 : 0
    );
  };

  // Function to format date
  const formatDate = (timestamp) => {
    if (!timestamp) return 'Unknown Date';
    return timestamp.slice(0, 10); // Assumes ISO 8601 format
  };

  return (
    <div>
      {/* Navigation Buttons */}
      <div className="flex justify-between mb-4">
        <button
          onClick={handlePrevious}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Previous
        </button>
        <span className="font-bold">
          Sequence {currentIndex + 1} of {totalSequences}
        </span>
        <button
          onClick={handleNext}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Next
        </button>
      </div>

      {/* Graph Title with Metadata */}
      <h2 className="text-center text-xl font-bold mb-4">
        {`Ticker: ${ticker}, Start: ${formatDate(
          startTimestamp
        )}, End: ${formatDate(endTimestamp)}`}
      </h2>

      {/* Multi-Line Graph */}
      <div
        className="sequence-multi-line grid gap-4"
        style={{
          gridTemplateColumns: `repeat(${numColumns}, minmax(0, 1fr))`,
        }}
      >
        {features.map((feature, index) => (
          <div key={index}>
            <Line data={featureDataMap[feature]} title={`Feature: ${feature}`} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default SequenceMultiLine;