import React, { useState, useRef, useEffect } from "react";
import Line from "./Line";

const LINE_LIMIT = 500; // Define constant for max lines

/**
 * MultiLine:
 *
 * Expects `visualization.data` to have the following structure:
 * {
 *   "x": [...],            // array of x-values
 *   "lines": [...],        // 3D array of shape (batch, seq_length, features)
 *   "shape": [batch, seq_length, features]
 * }
 *
 * Supports dynamic compression levels:
 *   - Level 0: All batch * features lines on one graph
 *   - Level 1: One graph per batch, all features on the same graph
 *   - Level 2: One graph per batch * feature pair
 */
const MultiLine = ({ visualization }) => {
  const [compressionLevel, setCompressionLevel] = useState(2); // Default to lowest compression (each line on its own graph)
  const [currentIndex, setCurrentIndex] = useState(0);
  const inputRef = useRef(null);

  const { data: { x = [], lines = [], shape = [] } = {}, config = {} } = visualization;

  const [batchSize, seqLength, numFeatures] = shape;

  // Ensure input box reflects current index
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.value = currentIndex + 1; // Show 1-based index
    }
  }, [currentIndex]);

  // Validate input data, and return early if invalid
  if (!batchSize || !seqLength || !numFeatures) {
    return <div>No line chart data available</div>;
  }

  // Transform data based on compression level with line limit
  let transformedLines = [];
  let isLimited = false;

  if (compressionLevel === 0) {
    // All batch * features lines on one graph
    transformedLines = Array.from({ length: batchSize }, (_, b) =>
      Array.from({ length: numFeatures }, (_, f) => ({
        label: `Batch ${b}, Feature ${f}`,
        values: lines[b].map((seq) => seq[f]),
      }))
    ).flat();

    // Apply line limit if needed
    if (transformedLines.length > LINE_LIMIT) {
      transformedLines = transformedLines.slice(0, LINE_LIMIT);
      isLimited = true;
    }

  } else if (compressionLevel === 1) {
    // One graph per batch, all features on the same graph
    transformedLines = Array.from({ length: batchSize }, (_, b) => ({
      label: `Batch ${b}`,
      values: Array.from({ length: numFeatures }, (_, f) => ({
        label: `Feature ${f}`,
        values: lines[b].map((seq) => seq[f]),
      })),
    }));

    // Check if any batch has too many features
    if (numFeatures > LINE_LIMIT) {
      transformedLines = transformedLines.map(batch => ({
        ...batch,
        values: batch.values.slice(0, LINE_LIMIT)
      }));
      isLimited = true;
    }

  } else if (compressionLevel === 2) {
    // One graph per batch * feature pair
    transformedLines = Array.from({ length: batchSize }, (_, b) =>
      Array.from({ length: numFeatures }, (_, f) => ({
        label: `Batch ${b}, Feature ${f}`,
        values: lines[b].map((seq) => seq[f]),
      }))
    ).flat();

    // Apply line limit if needed
    if (transformedLines.length > LINE_LIMIT) {
      transformedLines = transformedLines.slice(0, LINE_LIMIT);
      isLimited = true;
    }
  }

  const totalGraphs = transformedLines.length;

  // Handle navigation with buttons and text input
  const handleNext = () => setCurrentIndex((prev) => Math.min(prev + 1, totalGraphs - 1));
  const handlePrevious = () => setCurrentIndex((prev) => Math.max(prev - 1, 0));
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      let newIndex = parseInt(inputRef.current.value, 10);
      if (isNaN(newIndex)) newIndex = 1;
      newIndex = Math.max(1, Math.min(newIndex, totalGraphs));
      setCurrentIndex(newIndex - 1); // Convert to 0-based index
    }
  };

  // Handlers for compression level changes
  const handleCompressionChange = (direction) => {
    setCompressionLevel((prev) => Math.min(Math.max(prev + direction, 0), 2));
    setCurrentIndex(0); // Reset index when changing compression level
  };

  // Prepare the current graph data
  let currentGraph = {};
  if (compressionLevel === 0) {
    currentGraph = {
      lines: transformedLines,
      x,
    };
  } else if (compressionLevel === 1) {
    currentGraph = {
      lines: transformedLines[currentIndex].values,
      x,
    };
  } else if (compressionLevel === 2) {
    currentGraph = {
      lines: [transformedLines[currentIndex]],
      x,
    };
  }

  const currentConfig = {
    title:
      compressionLevel === 0
        ? config.title || "All Lines"
        : compressionLevel === 1
        ? `${transformedLines[currentIndex].label} (All Features)`
        : transformedLines[currentIndex].label,
    xAxisLabel: config.xAxisLabel || "X Axis",
    yAxisLabel: config.yAxisLabel || "Y Axis",
  };

  const lineVisualization = {
    data: currentGraph,
    config: currentConfig,
  };

  return (
    <div style={{ color: "#fff" }}>
      {/* Show warning if lines were limited */}
      {isLimited && (
        <div style={{ 
          backgroundColor: 'rgba(255, 165, 0, 0.2)', 
          color: 'orange', 
          padding: '0.5rem', 
          marginBottom: '1rem',
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          Warning: Only showing first {LINE_LIMIT} lines due to performance limitations
        </div>
      )}

      {/* Compression level controls */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "1rem" }}>
        <button onClick={() => handleCompressionChange(-1)} disabled={compressionLevel === 0}>
          Compress
        </button>
        <span style={{ margin: "0 1rem" }}>
          Compression Level:{" "}
          {compressionLevel === 0
            ? "All Lines"
            : compressionLevel === 1
            ? "Per Batch"
            : "Per Line"}
        </span>
        <button onClick={() => handleCompressionChange(1)} disabled={compressionLevel === 2}>
          Expand
        </button>
      </div>

      {/* Navigation row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "1rem",
        }}
      >
        <button onClick={handlePrevious} disabled={currentIndex === 0}>
          &larr;
        </button>

        {/* Input box for navigation */}
        <div style={{ display: "flex", alignItems: "center" }}>
          <span style={{ marginRight: "0.5rem" }}>Showing Graph</span>
          <input
            type="number"
            style={{ width: "3rem", textAlign: "center" }}
            defaultValue={currentIndex + 1}
            min="1"
            max={totalGraphs}
            onKeyDown={handleKeyDown}
            ref={inputRef}
          />
          <span style={{ marginLeft: "0.5rem" }}>of {totalGraphs}</span>
        </div>

        <button onClick={handleNext} disabled={currentIndex === totalGraphs - 1}>
          &rarr;
        </button>
      </div>

      {/* Render the <Line> component */}
      <Line visualization={lineVisualization} />
    </div>
  );
};

export default MultiLine;