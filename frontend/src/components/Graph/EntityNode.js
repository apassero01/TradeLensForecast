import React, { useMemo, useState } from 'react';
import { Handle, Position } from 'reactflow';
import MetadataList from './MetadataList';
import MetadataValue from './MetadataValue';
import visualizationComponents from '../Visualization/visualizationComponents';

const MIN_WIDTH = 250;
const MIN_HEIGHT = 100;

const EntityNode = React.memo(({ data }) => {
  const [dimensions, setDimensions] = useState({
    width: MIN_WIDTH,
    // Instead of 'auto', store a numeric height so flex can size properly:
    height: 150, // or whatever your starting height is
  });
  const [isResizing, setIsResizing] = useState(false);

  // Start the resize drag
  const handleResizeStart = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);

    const startX = e.clientX;
    const startY = e.clientY;
    const { width: startWidth, height: startHeight } = dimensions;

    const handleResizeMove = (moveEvent) => {
      moveEvent.preventDefault();
      moveEvent.stopPropagation();
      const newWidth = Math.max(
        MIN_WIDTH,
        startWidth + (moveEvent.clientX - startX)
      );
      const newHeight = Math.max(
        MIN_HEIGHT,
        startHeight + (moveEvent.clientY - startY)
      );
      setDimensions({ width: newWidth, height: newHeight });
    };

    const handleResizeEnd = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
    };

    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeEnd);
  };

  // Build your content
  const content = useMemo(() => {
    const { visualization } = data;
    if (visualization?.type) {
      const VisualizationComponent =
        visualizationComponents[visualization.type.toLowerCase()];
      if (!VisualizationComponent) {
        console.warn(`No visualization found for type: ${visualization.type}`);
        return null;
      }
      // Let the visualization fill its container
      return (
        <div className="w-full h-full">
          <VisualizationComponent visualization={visualization} />
        </div>
      );
    }

    // If no visualization, fall back to metadata
    const metadata = data.meta_data || data.metaData || {};
    return (
      <>
        {Object.entries(metadata).map(([key, value]) => (
          <div key={key} className="text-sm flex items-start gap-2">
            <span className="text-gray-400">{key}:</span>
            {Array.isArray(value) ? (
              <MetadataList items={value} />
            ) : (
              <MetadataValue value={value} />
            )}
          </div>
        ))}
      </>
    );
  }, [data]);

  const handleContextMenu = (e) => {
    e.preventDefault();
    navigator.clipboard.writeText(data.id).then(
      () => console.log('Entity ID copied to clipboard:', data.id),
      (err) => console.error('Failed to copy ID:', err),
    );
  };

  return (
    <div
      onContextMenu={handleContextMenu}
      title={`Right click to copy ID: ${data.id}`}
      // A flex container with a set width & height that the user can resize
      className={`
        relative bg-gray-800 border border-gray-700 rounded-lg
        ${isResizing ? 'cursor-nwse-resize select-none' : 'cursor-grab active:cursor-grabbing'}
        flex flex-col
      `}
      style={{
        width: dimensions.width,
        height: dimensions.height,
        minWidth: `${MIN_WIDTH}px`,
        minHeight: `${MIN_HEIGHT}px`,
      }}
    >
      {/* React Flow handle on top */}
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        style={{ background: '#4b5563' }}
      />

      {/* Header area */}
      <div className="px-4 py-2 border-b border-gray-700">
        <div className="text-white font-medium">
          {data.entity_name}
        </div>
      </div>

      {/* Main content flex-grow area 
          min-h-0 is crucial in Tailwind to let this area grow/shrink. */}
      <div className="flex-grow min-h-0 px-4 py-2">
        {content}
      </div>

      {/* React Flow handle on bottom */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        style={{ background: '#4b5563' }}
      />

      {/* Resize corner (nodrag) */}
      <div
        className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize nodrag"
        style={{
          background: 'linear-gradient(135deg, transparent 50%, #4b5563 50%)',
          borderBottomRightRadius: '0.5rem',
        }}
        onMouseDown={handleResizeStart}
      />
    </div>
  );
});

export default EntityNode;