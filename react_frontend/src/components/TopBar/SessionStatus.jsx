import React from 'react';

const SessionStatus = ({ isActive }) => {
  return (
    <div className="flex items-center">
      <span 
        className={`h-2 w-2 rounded-full mr-2 ${
          isActive ? 'bg-green-500' : 'bg-gray-500'
        }`}
      />
      <span className="text-sm text-gray-300">
        {isActive ? 'Session Active' : 'No Active Session'}
      </span>
    </div>
  );
};

export default SessionStatus; 