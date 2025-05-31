import React, { useRef, useEffect, useState, useCallback } from 'react';
import { IoCode, IoEye, IoSettings, IoChevronDown } from 'react-icons/io5';
import Editor from '@monaco-editor/react';

// Monaco Editor type definitions
interface Monaco {
  KeyMod: any;
  KeyCode: any;
}

interface ModernEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  theme?: 'light' | 'dark';
  fontSize?: number;
  readOnly?: boolean;
  minimap?: boolean;
  lineNumbers?: boolean;
  wordWrap?: boolean;
  autoDetectLanguage?: boolean;
  className?: string;
  height?: string | number;
}

const LANGUAGE_MAP: { [key: string]: string } = {
  js: 'javascript',
  jsx: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  py: 'python',
  json: 'json',
  html: 'html',
  htm: 'html',
  css: 'css',
  scss: 'scss',
  sass: 'sass',
  less: 'less',
  md: 'markdown',
  markdown: 'markdown',
  xml: 'xml',
  yaml: 'yaml',
  yml: 'yaml',
  sql: 'sql',
  sh: 'shell',
  bash: 'shell',
  dockerfile: 'dockerfile',
  php: 'php',
  java: 'java',
  cpp: 'cpp',
  c: 'c',
  cs: 'csharp',
  go: 'go',
  rust: 'rust',
  ruby: 'ruby',
  text: 'plaintext',
  txt: 'plaintext'
};

const SUPPORTED_LANGUAGES = [
  'javascript', 'typescript', 'python', 'json', 'html', 'css', 'markdown',
  'xml', 'yaml', 'sql', 'shell', 'php', 'java', 'cpp', 'csharp', 'go',
  'rust', 'ruby', 'plaintext'
];

// Language detection heuristics
const detectLanguage = (content: string): string => {
  if (!content || content.trim().length === 0) return 'plaintext';
  
  const firstLine = content.trim().split('\n')[0];
  const lines = content.trim().split('\n');
  
  // Check for shebangs
  if (firstLine.startsWith('#!')) {
    if (firstLine.includes('python')) return 'python';
    if (firstLine.includes('node') || firstLine.includes('javascript')) return 'javascript';
    if (firstLine.includes('bash') || firstLine.includes('sh')) return 'shell';
    return 'shell';
  }
  
  // JSON detection
  try {
    if ((content.trim().startsWith('{') && content.trim().endsWith('}')) ||
        (content.trim().startsWith('[') && content.trim().endsWith(']'))) {
      JSON.parse(content);
      return 'json';
    }
  } catch (e) {
    // Not JSON
  }
  
  // Markdown detection
  const markdownPatterns = [
    /^#+\s+.+/m,           // Headers
    /^\s*[-*+]\s+.+/m,     // Lists
    /\[.+\]\(.+\)/,        // Links
    /```[\s\S]*?```/,      // Code blocks
    /\*\*[^*]+\*\*|__[^_]+__/, // Bold
    /\*[^*]+\*|_[^_]+_/,   // Italic
    /^\s*>\s+.+/m,         // Blockquotes
    /^\s*\|.+\|/m          // Tables
  ];
  
  let markdownScore = 0;
  markdownPatterns.forEach(pattern => {
    if (pattern.test(content)) markdownScore++;
  });
  
  if (markdownScore >= 2) return 'markdown';
  
  // HTML detection
  if (/<\/?[a-z][\s\S]*>/i.test(content)) return 'html';
  
  // CSS detection
  if (/[#.]?[\w-]+\s*\{[\s\S]*?\}/.test(content)) return 'css';
  
  // Python detection
  const pythonKeywords = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'print(', 'len('];
  const pythonScore = pythonKeywords.filter(keyword => content.includes(keyword)).length;
  if (pythonScore >= 2) return 'python';
  
  // JavaScript/TypeScript detection
  const jsKeywords = ['function', 'const ', 'let ', 'var ', '=>', 'console.log', 'document.', 'window.'];
  const jsScore = jsKeywords.filter(keyword => content.includes(keyword)).length;
  const hasInterface = content.includes('interface ');
  const hasType = /:\s*(string|number|boolean|object)/.test(content);
  
  if (hasInterface || hasType) return 'typescript';
  if (jsScore >= 2) return 'javascript';
  
  // YAML detection
  if (/^[\w-]+:\s*.+/m.test(content) && !content.includes('{') && !content.includes('[')) {
    return 'yaml';
  }
  
  // SQL detection
  const sqlKeywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP'];
  const sqlScore = sqlKeywords.filter(keyword => 
    new RegExp(`\\b${keyword}\\b`, 'i').test(content)
  ).length;
  if (sqlScore >= 2) return 'sql';
  
  return 'plaintext';
};

const ModernEditor: React.FC<ModernEditorProps> = ({
  value,
  onChange,
  language: propLanguage,
  theme = 'dark',
  fontSize = 14,
  readOnly = false,
  minimap = false,
  lineNumbers = true,
  wordWrap = true,
  autoDetectLanguage = true,
  className = '',
  height = '100%'
}) => {
  const [detectedLanguage, setDetectedLanguage] = useState<string>('plaintext');
  const [manuallySelectedLanguage, setManuallySelectedLanguage] = useState<string | null>(null);
  const [showLanguageDropdown, setShowLanguageDropdown] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [editorSettings, setEditorSettings] = useState({
    minimap,
    lineNumbers,
    wordWrap,
    fontSize
  });
  const editorRef = useRef<any>(null);
  const dropdownRef = useRef<any>(null);
  const settingsRef = useRef<any>(null);

  // Auto-detect language when content changes (only if no manual selection)
  useEffect(() => {
    if (autoDetectLanguage && value && !manuallySelectedLanguage) {
      const detected = detectLanguage(value);
      setDetectedLanguage(detected);
    }
  }, [value, autoDetectLanguage, manuallySelectedLanguage]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowLanguageDropdown(false);
      }
      if (settingsRef.current && !settingsRef.current.contains(event.target)) {
        setShowSettings(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Determine current language with proper precedence
  const getCurrentLanguage = () => {
    // Manual selection should override everything except when explicitly disabled
    if (manuallySelectedLanguage) return manuallySelectedLanguage;
    if (propLanguage) return propLanguage;
    if (autoDetectLanguage) return detectedLanguage;
    return 'plaintext';
  };

  const currentLanguage = getCurrentLanguage();
  const normalizedLanguage = LANGUAGE_MAP[currentLanguage] || currentLanguage;

  console.log('Language states:', {
    propLanguage,
    manuallySelectedLanguage,
    detectedLanguage,
    currentLanguage,
    normalizedLanguage
  });

  const handleEditorChange = useCallback((newValue: string | undefined) => {
    onChange(newValue || '');
  }, [onChange]);

  const handleLanguageSelect = (selectedLanguage: string) => {
    console.log('Language selected:', selectedLanguage);
    setManuallySelectedLanguage(selectedLanguage);
    setShowLanguageDropdown(false);
    console.log('Manual language set to:', selectedLanguage);
  };

  const handleEditorDidMount = (editor: any, monaco: Monaco) => {
    editorRef.current = editor;
    
    // Configure editor options
    editor.updateOptions({
      fontSize: editorSettings.fontSize,
      minimap: { enabled: editorSettings.minimap },
      lineNumbers: editorSettings.lineNumbers ? 'on' : 'off',
      wordWrap: editorSettings.wordWrap ? 'on' : 'off',
      automaticLayout: true,
      scrollBeyondLastLine: false,
      renderWhitespace: 'selection',
      cursorBlinking: 'smooth',
      cursorSmoothCaretAnimation: 'on',
      smoothScrolling: true,
      contextmenu: true,
      folding: true,
      foldingHighlight: true,
      showFoldingControls: 'mouseover',
      bracketPairColorization: { enabled: true },
      guides: {
        bracketPairs: true,
        indentation: true
      }
    });

    // Add keyboard shortcuts
    editor.addAction({
      id: 'save-action',
      label: 'Save',
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS],
      run: () => {
        // Trigger save (could be handled by parent component)
        const event = new CustomEvent('editorSave', { detail: value });
        document.dispatchEvent(event);
      }
    });
  };

  const updateEditorSettings = (newSettings: Partial<typeof editorSettings>) => {
    const updated = { ...editorSettings, ...newSettings };
    setEditorSettings(updated);
    
    if (editorRef.current) {
      editorRef.current.updateOptions({
        fontSize: updated.fontSize,
        minimap: { enabled: updated.minimap },
        lineNumbers: updated.lineNumbers ? 'on' : 'off',
        wordWrap: updated.wordWrap ? 'on' : 'off',
      });
    }
  };

  return (
    <div 
      className={`nodrag flex flex-col h-full bg-gray-900 ${className}`}
      onMouseDown={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
      onTouchStart={(e) => e.stopPropagation()}
    >
      {/* Editor Toolbar */}
      <div 
        className="flex-shrink-0 bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between"
        onMouseDown={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-4">
          {/* Language Selector */}
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setShowLanguageDropdown(!showLanguageDropdown)}
              className="flex items-center gap-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white"
            >
              {IoCode({ size: 16 }) as React.JSX.Element}
              <span>{normalizedLanguage}</span>
              {IoChevronDown({ size: 14 }) as React.JSX.Element}
            </button>
            
            {showLanguageDropdown && (
              <div className="absolute top-full left-0 mt-1 bg-gray-700 border border-gray-600 rounded shadow-lg z-50 max-h-48 overflow-y-auto">
                {SUPPORTED_LANGUAGES.map(lang => (
                  <button
                    key={lang}
                    onClick={() => handleLanguageSelect(lang)}
                    className={`block w-full text-left px-3 py-2 text-sm hover:bg-gray-600 ${
                      normalizedLanguage === lang ? 'bg-blue-600 text-white' : 'text-gray-200'
                    }`}
                  >
                    {lang}
                  </button>
                ))}
                <div className="border-t border-gray-600 mt-1 pt-1">
                  <button
                    onClick={() => {
                      setManuallySelectedLanguage(null);
                      setShowLanguageDropdown(false);
                    }}
                    className="block w-full text-left px-3 py-2 text-sm hover:bg-gray-600 text-yellow-300"
                  >
                    🔄 Auto-detect
                  </button>
                </div>
              </div>
            )}
          </div>

          {autoDetectLanguage && !manuallySelectedLanguage && (
            <span className="text-xs text-gray-400">
              Auto-detected: {detectedLanguage}
            </span>
          )}

          {manuallySelectedLanguage && (
            <span className="text-xs text-blue-400">
              Manual: {manuallySelectedLanguage}
            </span>
          )}

          {/* Debug info - remove this after testing */}
          <span className="text-xs text-red-400">
            Debug: prop={propLanguage || 'null'}, manual={manuallySelectedLanguage || 'null'}, detected={detectedLanguage}, current={currentLanguage}
          </span>
        </div>

        {/* Settings */}
        <div className="relative" ref={settingsRef}>
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm text-white"
          >
            {IoSettings({ size: 16 }) as React.JSX.Element}
          </button>
          
          {showSettings && (
            <div className="absolute top-full right-0 mt-1 bg-gray-700 border border-gray-600 rounded shadow-lg z-50 p-4 min-w-48">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm text-white">Font Size</label>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => updateEditorSettings({ fontSize: Math.max(10, editorSettings.fontSize - 1) })}
                      className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs"
                    >
                      -
                    </button>
                    <span className="text-xs text-gray-300 min-w-[2rem] text-center">
                      {editorSettings.fontSize}
                    </span>
                    <button
                      onClick={() => updateEditorSettings({ fontSize: Math.min(24, editorSettings.fontSize + 1) })}
                      className="px-2 py-1 bg-gray-600 hover:bg-gray-500 rounded text-xs"
                    >
                      +
                    </button>
                  </div>
                </div>
                
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm text-white">Minimap</span>
                  <input
                    type="checkbox"
                    checked={editorSettings.minimap}
                    onChange={(e) => updateEditorSettings({ minimap: e.target.checked })}
                    className="rounded"
                  />
                </label>
                
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm text-white">Line Numbers</span>
                  <input
                    type="checkbox"
                    checked={editorSettings.lineNumbers}
                    onChange={(e) => updateEditorSettings({ lineNumbers: e.target.checked })}
                    className="rounded"
                  />
                </label>
                
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-sm text-white">Word Wrap</span>
                  <input
                    type="checkbox"
                    checked={editorSettings.wordWrap}
                    onChange={(e) => updateEditorSettings({ wordWrap: e.target.checked })}
                    className="rounded"
                  />
                </label>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Monaco Editor */}
      <div 
        className="flex-1 nodrag" 
        style={{ height: typeof height === 'string' ? height : `${height}px` }}
        onMouseDown={(e) => e.stopPropagation()}
        onPointerDown={(e) => e.stopPropagation()}
        onTouchStart={(e) => e.stopPropagation()}
      >
        <Editor
          height="100%"
          language={SUPPORTED_LANGUAGES.includes(normalizedLanguage) ? normalizedLanguage : 'plaintext'}
          theme={theme === 'dark' ? 'vs-dark' : 'vs'}
          value={value}
          onChange={handleEditorChange}
          onMount={handleEditorDidMount}
          options={{
            readOnly,
            fontSize: editorSettings.fontSize,
            minimap: { enabled: editorSettings.minimap },
            lineNumbers: editorSettings.lineNumbers ? 'on' : 'off',
            wordWrap: editorSettings.wordWrap ? 'on' : 'off',
            automaticLayout: true,
            scrollBeyondLastLine: false,
            renderWhitespace: 'selection',
            cursorBlinking: 'smooth',
            cursorSmoothCaretAnimation: 'on',
            smoothScrolling: true,
            contextmenu: true,
            folding: true,
            foldingHighlight: true,
            showFoldingControls: 'mouseover',
            bracketPairColorization: { enabled: true },
            guides: {
              bracketPairs: true,
              indentation: true
            },
            padding: { top: 16, bottom: 16 },
            suggestOnTriggerCharacters: true,
            quickSuggestions: true,
            parameterHints: { enabled: true },
            formatOnPaste: true,
            formatOnType: true
          }}
        />
      </div>
    </div>
  );
};

export default ModernEditor; 