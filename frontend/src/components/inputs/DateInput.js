// src/components/DateInput.js
import React from 'react';

function DateInput({label, onDateChange, selectedDate}) {
    const handleChange = (event) => {
        onDateChange(event.target.value);  // Call the onDateChange function passed from parent
    };

    return (
        <div>
            <label>
                {label}
                <input
                    type="date"
                    onChange={handleChange}
                    value={selectedDate || ''}  // Initialize with curDate if it's not null, otherwise empty
                />
            </label>
        </div>
    );
}

export default DateInput;