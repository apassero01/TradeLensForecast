// src/components/DateInput.js
import React from 'react';

function DateInput({label, onDateChange, selectedDate}) {
    const handleChange = (event) => {
        onDateChange(event.target.value);
    };

    return (
        <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
                {label}
            </label>
            <input
                type="date"
                onChange={handleChange}
                value={selectedDate || ''}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg 
                         text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                         placeholder-gray-400"
            />
        </div>
    );
}

export default DateInput;