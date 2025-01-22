// RequestEditorModal.js
import React, { useState, useEffect } from 'react';
import Editor from '../Visualization/Editor'; // or wherever your Ace Editor is
import Modal from '../Common/Modal'; // or a custom modal wrapper

const RequestEditorModal = ({
  request,
  onClose,
  onSave,      // Called when user saves the updated request JSON
  onExecute,   // Called when user wants to execute immediately
}) => {
  const [editorText, setEditorText] = useState('');

  useEffect(() => {
    if (request) {
      // Convert the request object to a JSON string
      const jsonStr = JSON.stringify(request, null, 2);
      setEditorText(jsonStr);
    }
  }, [request]);

  // Update local editor text whenever user types in the editor
  const handleEditorChange = (newValue) => {
    setEditorText(newValue);
  };

  const handleSaveClick = () => {
    try {
      const parsed = JSON.parse(editorText);
      onSave(parsed);
    } catch (err) {
      alert('Invalid JSON: ' + err.message);
    }
  };

  const handleExecuteClick = () => {
    try {
      const parsed = JSON.parse(editorText);
      onExecute(parsed);
    } catch (err) {
      alert('Invalid JSON: ' + err.message);
    }
  };

  if (!request) return null; // No request selected

  return (
    <Modal onClose={onClose}>
      <div className="bg-gray-800 p-4 rounded shadow-lg w-[80vw] h-[80vh] flex flex-col">
        <h2 className="text-white text-lg mb-2">Edit Strategy Request</h2>

        <div className="flex-grow min-h-0 border border-gray-600">
          <Editor
            visualization={{
              data: editorText,
              config: { type: 'json', title: 'Strategy Request JSON' }
            }}
            // For two-way binding, pass an onChange prop if your Editor supports it:
            onChange={handleEditorChange}
          />
        </div>

        <div className="mt-4 flex justify-end space-x-2">
          <button
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500"
            onClick={onClose}
          >
            Close
          </button>

          <button
            className="px-4 py-2 bg-blue-700 text-white rounded hover:bg-blue-600"
            onClick={handleSaveClick}
          >
            Save
          </button>

          <button
            className="px-4 py-2 bg-green-700 text-white rounded hover:bg-green-600"
            onClick={handleExecuteClick}
          >
            Execute
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default RequestEditorModal;