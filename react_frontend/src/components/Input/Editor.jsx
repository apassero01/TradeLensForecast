import React, { useState, useEffect, useRef } from 'react';
import AceEditor from 'react-ace';

// Import ace-builds correctly
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/theme-monokai';

// Configure ace
import ace from 'ace-builds';
ace.config.set('basePath', '/ace-builds');
ace.config.set('workerPath', '/ace-builds');

const Editor = ({ visualization, onChange }) => {
  const [editorText, setEditorText] = useState('');
  const [fontSize, setFontSize] = useState(14);
  const [editorMode, setEditorMode] = useState('text');
  const editorRef = useRef(null);

  const hasData = !!(visualization && visualization.data);

  useEffect(() => {
    if (hasData) {
      const { data } = visualization;
      const initialText = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
      setEditorText(initialText);

      const { config = {} } = visualization;
      const type = (config.type || 'text').toLowerCase();
      
      const modeMap = {
        py: 'python',
        python: 'python',
        json: 'json',
        txt: 'text',
        text: 'text',
      };
      setEditorMode(modeMap[type] || 'text');
    } else {
      setEditorText('');
      setEditorMode('text');
    }
  }, [visualization, hasData]);

  if (!hasData) {
    return <div className="text-red-500">No editor data available</div>;
  }

  const { config = {} } = visualization;

  const handleFontSize = (change) => {
    setFontSize((prev) => Math.max(8, Math.min(24, prev + change)));
  };

  const handleModeChange = (e) => {
    setEditorMode(e.target.value);
  };

  const handleEditorChange = (newText) => {
    setEditorText(newText);
    if (onChange) {
      onChange(newText);
    }
  };

  const handleClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div  
      className="flex flex-col w-full h-full nodrag bg-gray-800"
      onClick={handleClick}
    >
      <div 
        className="flex-none border-b border-gray-700 p-2 flex justify-between items-center"
        onClick={handleClick}
      >
        <div className="flex items-center space-x-4">
          <h2 className="text-gray-200 text-lg">
            {config.title || 'Document Editor'}
          </h2>
          <select
            className="bg-gray-700 text-gray-200 p-1 rounded"
            value={editorMode}
            onChange={handleModeChange}
            onClick={handleClick}
          >
            <option value="python">Python</option>
            <option value="json">JSON</option>
            <option value="text">Text</option>
          </select>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleFontSize(-2);
            }}
            className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600"
          >
            A-
          </button>
          <span className="text-gray-300">{fontSize}px</span>
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleFontSize(2);
            }}
            className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600"
          >
            A+
          </button>
        </div>
      </div>

      <div 
        className="flex-grow min-h-0 relative"
        onClick={handleClick}
      >
        <AceEditor
          ref={editorRef}
          mode={editorMode}
          theme="monokai"
          value={editorText}
          onChange={handleEditorChange}
          fontSize={fontSize}
          width="100%"
          height="100%"
          name="document-editor"
          editorProps={{ 
            $blockScrolling: true,
          }}
          setOptions={{
            showLineNumbers: true,
            wrap: false,
            tabSize: 2,
            useSoftTabs: true,
            showPrintMargin: false,
            highlightActiveLine: false,
            useWorker: true,
          }}
          className="w-full h-full"
          onLoad={(editor) => {
            editor.container.style.transform = 'none';
            editor.container.addEventListener('click', handleClick);
          }}
          onClick={handleClick}
        />
      </div>
    </div>
  );
};

export default Editor;