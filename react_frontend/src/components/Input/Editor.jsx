import React, { useState, useEffect, useRef, useCallback } from 'react';
import AceEditor from 'react-ace';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

// Import ace-builds correctly
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/mode-json';
import 'ace-builds/src-noconflict/mode-markdown';
import 'ace-builds/src-noconflict/theme-monokai';

// Configure ace
import ace from 'ace-builds';
ace.config.set('basePath', '/ace-builds');
ace.config.set('workerPath', '/ace-builds');

// Helper function to guess if text is Markdown (simple heuristic)
const isLikelyMarkdown = (text) => {
  if (!text || typeof text !== 'string') return false;
  const  trimmedText = text.trim();

  // Requiring a minimum length to reduce false positives on short, generic texts.
  if (trimmedText.length < 20) return false;

  const markdownPatterns = [
    /^#+\s+.*/m,        // Headings (e.g., # Title)
    /^\s*[-*+]\s+.*/m,  // List items (e.g., - item)
    /\[.*?\]\(.*?\)/,   // Links (e.g., [text](url))
    /`{3}[\s\S]*?`{3}/, // Code blocks (e.g., ```code```)
    /\*\*[^*]+\*\*|__[^*]+__/, // Bold (e.g., **bold**)
    /\*[^*]+\*|_.*?_/,     // Italic (e.g., *italic*) (simplified to avoid conflict with __)
  ];

  let patternsFound = 0;
  for (const pattern of markdownPatterns) {
    if (pattern.test(trimmedText)) {
      patternsFound++;
    }
  }
  // If at least 2 different markdown patterns are found, consider it Markdown.
  // This threshold can be adjusted.
  return patternsFound >= 2;
};

const Editor = ({ visualization, onChange, sendStrategyRequest, data, viewEntityId, parentEntityId}) => {
  const [editorText, setEditorText] = useState(null);
  const [fontSize, setFontSize] = useState(14);
  const [editorMode, setEditorMode] = useState('text');
  const [isInitialized, setIsInitialized] = useState(false);
  const editorRef = useRef(null);

  const hasData = !!(visualization && visualization.data);



  // Define all callbacks at the top level, before any conditional returns
  const handleFontSize = useCallback((change) => {
    setFontSize((prev) => Math.max(8, Math.min(24, prev + change)));
  }, []);

  const handleModeChange = useCallback((e) => {
    setEditorMode(e.target.value);
  }, []);

  const handleEditorChange = useCallback((newText) => {
    setEditorText(newText);
    if (onChange) {
      onChange(newText);
    }
  }, [onChange]);

  const handleClick = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  // Add new save handler function
  const handleSave = useCallback(async () => {
    console.log('Save triggered. Current editorText:', editorText);
    
    if (!sendStrategyRequest) {
      console.error('Cannot save: sendStrategyRequest function is not available');
      return;
    }
    
    if (!visualization || !viewEntityId || !parentEntityId) {
      console.error('Cannot save: Missing required data (visualization, entityId, or parent_ids)');
      return;
    }
    
    // Check if editorText is defined before proceeding
    if (editorText === undefined || editorText === null) {
      console.error('Cannot save: Editor text is undefined');
      return;
    }
    
    try {
      const strategyRequest = {
        strategy_name: 'SetAttributesStrategy',
        param_config: {
          attribute_map: {
            'text': editorText,
          }
        },
        target_entity_id: parentEntityId,
        add_to_history: false,
        nested_requests: [],
      };
      
      await sendStrategyRequest(strategyRequest);
      console.log('Saved editor text to parent entity');
    } catch (error) {
      console.error('Failed to save editor text:', error);
    }
  }, [visualization, editorText, sendStrategyRequest, parentEntityId]);

  // Only initialize the editor content once or when visualization.data changes
  useEffect(() => {
    // Skip if already initialized or no data available
    if (isInitialized || !hasData) return;
    
    let initialText = '';
    
    // Check if visualization.data is a string or an object
    if (typeof visualization.data === 'string') {
      initialText = visualization.data;
    } else if (typeof visualization.data === 'object' && visualization.data !== null) {
      if (visualization.data.text !== undefined) {
        initialText = visualization.data.text;
      } else {
        try {
          initialText = JSON.stringify(visualization.data, null, 2);
        } catch (e) {
          console.error('Failed to stringify visualization data:', e);
          initialText = ''; // Fallback to empty string on error
        }
      }
    }
    
    setEditorText(initialText);

    const { config = {} } = visualization;
    let initialMode = (config.type || '').toLowerCase();
    
    const modeMap = {
      py: 'python',
      python: 'python',
      json: 'json',
      txt: 'text',
      text: 'text',
      md: 'markdown',
      markdown: 'markdown',
    };

    if (modeMap[initialMode]) {
      setEditorMode(modeMap[initialMode]);
    } else if (isLikelyMarkdown(initialText)) {
      setEditorMode('markdown');
    } else {
      setEditorMode('text'); // Default to text if no type and not detected as Markdown
    }
    
    // Special handling for text editor with object data that has a text attribute
    // This needs to be re-evaluated in the context of auto-detection
    if (editorMode === 'text' && // Check current editorMode, not initialMode
        typeof visualization.data === 'object' && 
        visualization.data !== null &&
        visualization.data.text !== undefined) {
      // If mode ended up as text, but data structure suggests specific text field, use it.
      // This might override JSON view if data is an object with a 'text' field but not explicit 'text' or 'markdown' type.
      setEditorText(visualization.data.text);
    }
    
    setIsInitialized(true);  // Mark as initialized
  }, [visualization, hasData, isInitialized]);

  // Only reset the editor when visualization.data actually changes
  // We use JSON.stringify to detect actual data changes
  const dataSignature = JSON.stringify(visualization?.data);
  
  useEffect(() => {
    // Skip if not initialized yet (let the initialization effect handle it)
    if (!isInitialized) return;
    
    // Reset the editor only when visualization.data actually changes
    setIsInitialized(false);  // Force re-initialization
  }, [dataSignature]);  // Only run when the data signature changes

  if (!hasData) {
    return <div className="text-red-500">No editor data available</div>;
  }

  const { config = {} } = visualization;

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
            <option value="markdown">Markdown</option>
          </select>
        </div>
        <div className="flex items-center space-x-2">
          {/* Add Save button */}
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              handleSave();
            }}
            className={`px-3 py-1 rounded text-white mr-4 ${
              sendStrategyRequest ? 'bg-green-600 hover:bg-green-500' : 'bg-gray-500 cursor-not-allowed'
            }`}
            disabled={!sendStrategyRequest}
            title={!sendStrategyRequest ? 'Save functionality not available' : 'Save document'}
          >
            Save
          </button>
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
        {editorMode === 'markdown' ? (
          <div 
            className="p-4 prose prose-invert max-w-none overflow-auto h-full nowheel"
            style={{ fontSize: `${fontSize}px` }}
            onWheel={(e) => e.stopPropagation()}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {editorText || ''}
            </ReactMarkdown>
          </div>
        ) : (
          <AceEditor
            ref={editorRef}
            mode={editorMode}
            theme="monokai"
            value={editorText === null ? '' : editorText}
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
            readOnly={config.readOnly || false}
            className="w-full h-full"
            onLoad={(editor) => {
              editor.container.style.transform = 'none';
              editor.container.addEventListener('click', handleClick);
            }}
            onClick={handleClick}
          />
        )}
      </div>
    </div>
  );
};

export default Editor;