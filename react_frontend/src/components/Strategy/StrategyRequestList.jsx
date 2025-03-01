import React from 'react';
import { FaPlay, FaEdit } from 'react-icons/fa';

function StrategyRequestList({ data }) {
  // We'll assume data.childrenRequests is an array of objects like:
  // [
  //   { id: 'req-1', strategy_name: 'Strategy 1', label: 'Child 1' },
  //   { id: 'req-2', strategy_name: 'Strategy 2', label: 'Child 2' },
  //   ...
  // ]
  const { childrenRequests = [] } = data;

  // Placeholder handlers
  const handlePlayClick = (request) => {
    // TODO: define what happens when user clicks the play button
    console.log('Play clicked for:', request);
  };

  const handleEditClick = (request) => {
    // TODO: define what happens when user clicks the edit button
    console.log('Edit clicked for:', request);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid transparent', // transparent border
        padding: '8px',
      }}
    >
      {childrenRequests.map((request) => (
        <div
          key={request.id}
          style={{
            display: 'flex',
            alignItems: 'center',
            marginBottom: '4px',
          }}
        >
          {/* Play Button */}
          <button
            onClick={() => handlePlayClick(request)}
            style={{
              backgroundColor: 'green',
              color: 'white',
              border: 'none',
              padding: '4px 8px',
              cursor: 'pointer',
            }}
            title={request.strategy_name} // Show strategy_name on hover
          >
            <FaPlay />
          </button>

          {/* Label or child info */}
          <div style={{ flex: 1, marginLeft: '8px' }}>
            {request.label || request.strategy_name}
          </div>

          {/* Edit Button */}
          <button
            onClick={() => handleEditClick(request)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            <FaEdit />
          </button>
        </div>
      ))}
    </div>
  );
}

export default StrategyRequestList;