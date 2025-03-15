import React, { useState, useRef, useEffect } from 'react';

const MetadataValue = ({ value }) => {
  const [showCopyButton, setShowCopyButton] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  const handleRightClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = containerRef.current.getBoundingClientRect();
    setPosition({ 
      x: e.clientX - rect.left, 
      y: e.clientY - rect.top 
    });
    setShowCopyButton(true);
  };

  const handleCopy = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(value.toString());
    setShowCopyButton(false);
  };

  // Close popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showCopyButton && !e.target.closest('.copy-popup')) {
        setShowCopyButton(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [showCopyButton]);

  return (
    <span 
      ref={containerRef}
      className="text-gray-400 bg-gray-700/50 px-1 py-0.5 rounded hover:bg-gray-600/50 cursor-context-menu relative"
      onContextMenu={handleRightClick}
      title="Right click for copy option"
    >
      {value}
      {showCopyButton && (
        <div 
          className="copy-popup absolute z-10 bg-gray-700 rounded shadow-lg py-1 px-2 text-xs"
          style={{ 
            left: `${position.x}px`, 
            top: `${position.y}px`,
            transform: 'translate(-50%, -100%)',
            marginTop: '-8px'
          }}
        >
          <button
            onClick={handleCopy}
            className="text-white hover:text-blue-300 transition-colors"
          >
            Copy
          </button>
        </div>
      )}
    </span>
  );
};

export default React.memo(MetadataValue); 