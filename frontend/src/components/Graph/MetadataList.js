import React, { useState, useRef, useEffect } from 'react';

const MetadataList = ({ items }) => {
  const [showCopyButton, setShowCopyButton] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);
  
  const renderItem = (item, index) => {
    const isNumber = typeof item === 'number';
    return (
      <React.Fragment key={index}>
        <span className="text-blue-300">
          {isNumber ? item : `"${item}"`}
        </span>
        {index < items.length - 1 && <span className="text-gray-500">, </span>}
      </React.Fragment>
    );
  };

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
    const formattedList = items.map(item => 
      typeof item === 'number' ? item : `"${item}"`
    ).join(', ');
    navigator.clipboard.writeText(`[${formattedList}]`);
    setShowCopyButton(false);
  };

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (showCopyButton && !e.target.closest('.copy-popup')) {
        setShowCopyButton(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [showCopyButton]);

  if (!Array.isArray(items)) return null;
  
  return (
    <div className="relative max-w-[200px] group" ref={containerRef}>
      <div className="relative">
        <div 
          className="overflow-x-auto scrollbar-hide whitespace-nowrap"
          onContextMenu={handleRightClick}
          title="Right click for copy option"
        >
          <div className="inline-block text-gray-400 bg-gray-700/50 px-1 py-0.5 rounded hover:bg-gray-600/50 cursor-context-menu">
            [{items.map(renderItem)}]
          </div>
        </div>
        <div className="absolute inset-y-0 left-0 w-4 bg-gradient-to-r from-gray-800 to-transparent pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity" />
        <div className="absolute inset-y-0 right-0 w-4 bg-gradient-to-l from-gray-800 to-transparent pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
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
    </div>
  );
};

export default React.memo(MetadataList); 