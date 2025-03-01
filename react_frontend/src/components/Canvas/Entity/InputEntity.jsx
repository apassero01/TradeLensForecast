// src/components/Canvas/InputEntity.jsx
import React, { memo, useState, useEffect } from 'react';
import EntityNodeBase from './EntityNodeBase';

function InputEntity({ data }) {
  const [text, setText] = useState(data.visualization || '');

  useEffect(() => {
    console.log('Updated text state:', text);
  }, [text]);

  const handleChange = (e, updateLocalField) => {
    const newValue = e.target.value;
    // Update both the local state and the parent's local field.
    setText(newValue);
    updateLocalField('visualization', newValue);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Perform any submission logic here.
  };

  return (
    <EntityNodeBase data={data}>
      {({ updateLocalField }) => (
        <form onSubmit={handleSubmit} className="w-full">
          <textarea
            value={text}
            onChange={(e) => handleChange(e, updateLocalField)}
            placeholder="Enter text..."
            className="w-full h-24 p-2 bg-gray-700 text-white rounded"
            style={{ resize: 'none' }}
          />
          <button
            type="submit"
            className="mt-2 w-full px-2 py-1 bg-gray-600 hover:bg-gray-500 text-white rounded"
          >
            Submit
          </button>
        </form>
      )}
    </EntityNodeBase>
  );
}

export default memo(InputEntity);