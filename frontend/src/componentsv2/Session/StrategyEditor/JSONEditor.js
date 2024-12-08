import React from 'react';

const JSONEditor = ({ value, onChange }) => {
  return (
    <div className="bg-gray-900 rounded p-4 h-[200px]">
      <textarea
        className="w-full h-full bg-gray-900 text-white p-2 font-mono resize-none
                 focus:outline-none focus:ring-1 focus:ring-blue-500
                 border border-gray-700 rounded"
        value={JSON.stringify(value, null, 2)}
        onChange={(e) => {
          try {
            const newValue = JSON.parse(e.target.value);
            onChange(newValue);
          } catch (error) {
            // Handle invalid JSON
            console.log('Invalid JSON');
          }
        }}
        spellCheck="false"
      />
    </div>
  );
};

export default JSONEditor; 