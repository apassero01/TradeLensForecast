import React, { useState } from 'react';

function JSONEditorModal({ initialConfig, onSave, onCancel }) {
    const [jsonValue, setJsonValue] = useState(JSON.stringify(initialConfig, null, 2));

    const handleSave = () => {
        try {
            const parsedConfig = JSON.parse(jsonValue); // Ensure valid JSON
            onSave(parsedConfig); // Pass the parsed config back to the parent component
        } catch (error) {
            alert('Invalid JSON format');
        }
    };

    return (
        <div
            className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50"
            onClick={(e) => e.stopPropagation()} // Prevent click propagation to the card
        >
            <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg w-1/2 h-3/4 flex flex-col" onClick={(e) => e.stopPropagation()}>
                <h2 className="text-xl font-semibold mb-4">Edit Strategy Configuration</h2>

                {/* Flex-grow Textarea */}
                <div className="flex-grow mb-4">
                    <textarea
                        className="w-full h-full p-2 border border-gray-700 rounded-lg bg-gray-900 text-green-400 font-mono resize-none"
                        value={jsonValue}
                        onChange={(e) => setJsonValue(e.target.value)}
                    />
                </div>

                <div className="mt-4 flex justify-center space-x-2 ju">
                    <button className="bg-green-500 text-white px-4 py-2 rounded-lg" onClick={handleSave}>
                        Save
                    </button>
                    <button className="bg-gray-500 text-white px-4 py-2 rounded-lg" onClick={onCancel}>
                        Cancel
                    </button>
                </div>
            </div>
        </div>
    );
}

export default JSONEditorModal;