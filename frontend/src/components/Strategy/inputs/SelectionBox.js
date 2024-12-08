import React, {useState} from 'react';
import '../../../index.css';

function SelectionBox({label, items, itemKey, displayText, onSelectionChange, selectedItems}) {
    // Create state to store the search query
    const [searchQuery, setSearchQuery] = useState('');

    const handleChange = (event, item) => {
        const isSelected = event.target.checked;
        onSelectionChange(item, isSelected);
    };

    // Filter items based on the search query
    const filteredItems = items.filter(item =>
        displayText(item).toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="space-y-4">
            <h3 className="text-lg font-semibold">{label}</h3>

            {/* Search input field */}
            <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search..."
                className="mb-4 p-2 border border-gray-300 rounded-lg w-full"
            />

            <form className="max-h-48 overflow-y-auto border border-gray-300 p-4 rounded-lg">
                {filteredItems.length > 0 ? (
                    filteredItems.map(item => (
                        <div key={item[itemKey]} className="mb-0">
                            <label className="flex items-center space-x-2">
                                <input
                                    type="checkbox"
                                    checked={selectedItems.some(selected => selected[itemKey] === item[itemKey])} // Determine if it's selected
                                    onChange={(event) => handleChange(event, item)}
                                    className="form-checkbox h-4 w-4 text-blue-600"
                                />
                                <span>{displayText(item)}</span>
                            </label>
                        </div>
                    ))
                ) : (
                    <p className="text-gray-500">No items found</p>
                )}
            </form>
        </div>
    );
}

export default SelectionBox;