import React, { useState, useEffect } from 'react';
import AceEditor from 'react-ace';

// Import ace editor themes and modes
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-text';
import 'ace-builds/src-noconflict/theme-monokai';
import 'ace-builds/src-noconflict/ext-language_tools';

const Editor = ({ visualization }) => {
  // 1) Always call Hooks at the top (unconditionally).
  const [editorText, setEditorText] = useState('');
  const [fontSize, setFontSize] = useState(14);
  const [editorMode, setEditorMode] = useState('text');

  // 2) Figure out whether we have data
  const hasData = !!(visualization && visualization.data);

  // 3) useEffect to load text only if we have valid data
  useEffect(() => {
    if (hasData) {
      const { data } = visualization;
      const initialText =
        typeof data === 'string'
          ? data
          : JSON.stringify(data, null, 2);
      setEditorText(initialText);

      const { config = {} } = visualization;
      const type = (config.type || 'text').toLowerCase();
      
      // Map recognized modes
      const modeMap = {
        py: 'python',
        python: 'python',
        json: 'json',
        txt: 'text',
        text: 'text',
      };
      setEditorMode(modeMap[type] || 'text');
    } else {
      // If no data, reset text/mode
      setEditorText('');
      setEditorMode('text');
    }
  }, [visualization, hasData]);

  // 4) If there's no data, show a message -- AFTER the Hooks.
  if (!hasData) {
    return <div className="text-red-500">No editor data available</div>;
  }

  // 5) If we do have data, render the editor
  const { config = {} } = visualization;

  const handleFontSize = (change) => {
    setFontSize((prev) => Math.max(8, Math.min(24, prev + change)));
  };

  const handleModeChange = (e) => {
    setEditorMode(e.target.value);
  };

  return (
    <div className="flex flex-col w-full h-full nodrag bg-gray-800">
      <div className="flex-none border-b border-gray-700 p-2 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <h2 className="text-gray-200 text-lg">
            {config.title || 'Document Editor'}
          </h2>
          {/* Dropdown for selecting the editor mode */}
          <select
            className="bg-gray-700 text-gray-200 p-1 rounded"
            value={editorMode}
            onChange={handleModeChange}
          >
            <option value="python">Python</option>
            <option value="json">JSON</option>
            <option value="text">Text</option>
          </select>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => handleFontSize(-2)}
            className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600"
          >
            A-
          </button>
          <span className="text-gray-300">{fontSize}px</span>
          <button
            onClick={() => handleFontSize(2)}
            className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600"
          >
            A+
          </button>
        </div>
      </div>

      <div className="flex-grow min-h-0">
        <AceEditor
          mode={editorMode}
          theme="monokai"
          value={editorText}
          onChange={setEditorText}
          fontSize={fontSize}
          width="100%"
          height="100%"
          name="document-editor"
          editorProps={{ $blockScrolling: true }}
          setOptions={{
            showLineNumbers: true,
            wrap: false,
            tabSize: 2,
            useSoftTabs: true,
            showPrintMargin: false,
            highlightActiveLine: false,
            enableBasicAutocompletion: false,
            enableLiveAutocompletion: false,
            useWorker: true,
          }}
          className="w-full h-full"
        />
      </div>
    </div>
  );
};

export default Editor;