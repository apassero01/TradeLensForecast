import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { IoSave, IoEye, IoCode, IoDocumentText } from 'react-icons/io5';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import ModernEditor from '../../../../../Input/ModernEditor';

interface AdvancedDocumentEditorProps {
  data?: AdvancedDocumentEditorData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface AdvancedDocumentEditorData {
  text?: string;
  file_type?: string;
  docName?: string;
  name?: string;
}

export default function AdvancedDocumentEditor({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: AdvancedDocumentEditorProps) {
  const [content, setContent] = useState(data?.text || '');
  const [isDirty, setIsDirty] = useState(false);
  const [fontSize, setFontSize] = useState(14);
  const editorRef = useRef<any>(null);

  // Helper function to detect markdown content (moved up for early use)
  const isContentMarkdown = (text: string): boolean => {
    if (!text || text.length < 5) return false;
    
    const markdownPatterns = [
      /^#+\s+.*/m,           // Headers
      /^\s*[-*+]\s+.*/m,     // Lists
      /\[.+\]\(.+\)/,        // Links
      /```[\s\S]*?```/,      // Code blocks
      /\*\*[^*]+\*\*|__[^_]+__/, // Bold
      /^\s*>\s+.*/m,         // Blockquotes
      /^\s*\|.+\|/m,         // Tables
      /\*[^*]+\*/,           // Italic
      /~~[^~]+~~/,           // Strikethrough
      /`[^`]+`/              // Inline code
    ];
    
    let matches = 0;
    for (const pattern of markdownPatterns) {
      if (pattern.test(text)) {
        matches++;
      }
    }
    
    return matches >= 1; // If 1+ markdown patterns found
  };

  // Initialize showPreview based on markdown detection
  const initialContentIsMarkdown = content && isContentMarkdown(content);
  const [showPreview, setShowPreview] = useState(initialContentIsMarkdown);
  const [hasManuallyToggled, setHasManuallyToggled] = useState(false);

  // Update content when data changes
  useEffect(() => {
    if (data?.text !== undefined && data.text !== content) {
      setContent(data.text);
      setIsDirty(false);
    }
  }, [data?.text]);

  // Auto-switch to preview when content becomes markdown (only if not manually toggled)
  useEffect(() => {
    const contentIsMarkdown = content && isContentMarkdown(content);
    if (contentIsMarkdown && !showPreview && !hasManuallyToggled) {
      setShowPreview(true);
    }
  }, [content, hasManuallyToggled, showPreview]);

  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    setIsDirty(newContent !== data?.text);
  };

  const handleSave = useCallback(() => {
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(parentEntityId)
      .withParams({
        attribute_map: { text: content }
      })
      .build());
    
    setIsDirty(false);
  }, [sendStrategyRequest, parentEntityId, content]);

  // Listen for editor save events
  useEffect(() => {
    const handleEditorSave = () => {
      handleSave();
    };

    document.addEventListener('editorSave', handleEditorSave);
    return () => document.removeEventListener('editorSave', handleEditorSave);
  }, [handleSave]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Save on Ctrl+S or Cmd+S
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      handleSave();
    }
  };



  const getEditorLanguage = () => {
    const fileType = data?.file_type || 'text';
    const languageMap: { [key: string]: string } = {
      python: 'python',
      py: 'python',
      javascript: 'javascript',
      js: 'javascript',
      typescript: 'typescript',
      ts: 'typescript',
      json: 'json',
      html: 'html',
      css: 'css',
      markdown: 'markdown',
      md: 'markdown',
      yaml: 'yaml',
      yml: 'yaml',
      sql: 'sql',
      shell: 'shell',
      sh: 'shell',
      bash: 'shell',
      text: 'plaintext',
      txt: 'plaintext'
    };
    return languageMap[fileType.toLowerCase()] || 'plaintext';
  };

  // More comprehensive markdown detection
  const editorLanguage = getEditorLanguage();
  const contentIsMarkdown = content && isContentMarkdown(content);
  
  // Force markdown detection if content has markdown patterns
  // This overrides the file_type check since the editor is clearly detecting markdown
  const isMarkdown = contentIsMarkdown ||
                     data?.file_type === 'markdown' || 
                     data?.file_type === 'md' || 
                     editorLanguage === 'markdown' ||
                     (data?.name && (data.name.endsWith('.md') || data.name.endsWith('.markdown'))) ||
                     (data?.docName && (data.docName.endsWith('.md') || data.docName.endsWith('.markdown')));

  // Debug log to see what's happening
  console.log('Markdown Detection Debug:', {
    'data.file_type': data?.file_type,
    'editorLanguage': editorLanguage,
    'contentIsMarkdown': contentIsMarkdown,
    'isMarkdown': isMarkdown,
    'showPreview': showPreview,
    'data.name': data?.name,
    'data.docName': data?.docName
  });

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No document selected
      </div>
    );
  }

  const displayName = data.docName || data.name || 'Untitled';

  return (
    <div className="flex flex-col h-full w-full bg-gray-900 nodrag">
      {/* Header */}
      <div className="flex-shrink-0 p-3 border-b border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {IoDocumentText({ className: "text-gray-400" }) as React.JSX.Element}
          <div>
            <h2 className="text-lg font-semibold text-white">
              {displayName}
              {data.docName && data.name && (
                <span className="text-gray-500 ml-2 text-sm font-normal">
                  ({data.name})
                </span>
              )}
            </h2>
            <p className="text-xs text-gray-500">
              {data.file_type || 'text'} • {content.length} characters
              {isDirty && <span className="text-yellow-400 ml-2">• Modified</span>}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Font Size Controls */}
          <div className="flex items-center gap-1 mr-2">
            <button
              onClick={() => setFontSize(Math.max(10, fontSize - 1))}
              className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
            >
              A-
            </button>
            <span className="text-xs text-gray-400 min-w-[2rem] text-center">
              {fontSize}px
            </span>
            <button
              onClick={() => setFontSize(Math.min(24, fontSize + 1))}
              className="px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
            >
              A+
            </button>
          </div>

          {/* Markdown Preview Toggle */}
          {isMarkdown && (
            <button
              onClick={() => {
                setShowPreview(!showPreview);
                setHasManuallyToggled(true);
              }}
              className={`px-3 py-1 rounded flex items-center gap-2 text-sm ${
                showPreview 
                  ? 'bg-blue-600 hover:bg-blue-700' 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {showPreview ? (
                <>
                  {IoCode({ size: 16 }) as React.JSX.Element} Edit
                </>
              ) : (
                <>
                  {IoEye({ size: 16 }) as React.JSX.Element} Preview
                </>
              )}
            </button>
          )}

          {/* Save Button */}
          <button
            onClick={handleSave}
            disabled={!isDirty}
            className={`px-3 py-1 rounded flex items-center gap-2 text-sm ${
              isDirty 
                ? 'bg-green-600 hover:bg-green-700' 
                : 'bg-gray-700 cursor-not-allowed opacity-50'
            }`}
          >
            {IoSave({ size: 16 }) as React.JSX.Element} Save
          </button>
        </div>
      </div>

      {/* Editor/Preview Area */}
      <div className="flex-1 flex flex-col" onKeyDown={handleKeyDown}>
        {showPreview && isMarkdown ? (
          <div 
            className="flex-1 overflow-y-auto overflow-x-hidden p-6 bg-gray-800 text-white" 
            style={{ 
              fontSize: `${fontSize}px`,
              minHeight: 0 
            }}
          >
            <div className="prose prose-invert max-w-none nowheel">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm, remarkBreaks]}
                components={{
                  // Custom styling for dark theme to ensure proper rendering
                  h1: ({...props}) => <h1 className="text-white text-3xl font-bold mb-4 border-b border-gray-600 pb-2" {...props} />,
                  h2: ({...props}) => <h2 className="text-white text-2xl font-semibold mb-3 border-b border-gray-700 pb-1" {...props} />,
                  h3: ({...props}) => <h3 className="text-white text-xl font-semibold mb-2" {...props} />,
                  h4: ({...props}) => <h4 className="text-white text-lg font-semibold mb-2" {...props} />,
                  h5: ({...props}) => <h5 className="text-white text-base font-semibold mb-2" {...props} />,
                  h6: ({...props}) => <h6 className="text-white text-sm font-semibold mb-2" {...props} />,
                  p: ({...props}) => <p className="text-gray-200 mb-4 leading-relaxed" {...props} />,
                  code: ({...props}) => <code className="bg-gray-700 text-blue-300 px-1 py-0.5 rounded text-sm" {...props} />,
                  pre: ({...props}) => <pre className="bg-gray-900 text-gray-200 p-4 rounded-lg overflow-x-auto mb-4 border border-gray-700" {...props} />,
                  blockquote: ({...props}) => <blockquote className="border-l-4 border-blue-500 pl-4 italic text-gray-300 mb-4 bg-gray-800 py-2" {...props} />,
                  ul: ({...props}) => <ul className="text-gray-200 mb-4 ml-6 list-disc" {...props} />,
                  ol: ({...props}) => <ol className="text-gray-200 mb-4 ml-6 list-decimal" {...props} />,
                  li: ({...props}) => <li className="mb-1" {...props} />,
                  a: ({...props}) => <a className="text-blue-400 hover:text-blue-300 underline" {...props} />,
                  table: ({...props}) => <table className="w-full border-collapse border border-gray-600 mb-4" {...props} />,
                  thead: ({...props}) => <thead className="bg-gray-700" {...props} />,
                  th: ({...props}) => <th className="border border-gray-600 text-white p-2 text-left" {...props} />,
                  td: ({...props}) => <td className="border border-gray-600 text-gray-200 p-2" {...props} />,
                  hr: ({...props}) => <hr className="border-gray-600 my-6" {...props} />,
                  strong: ({...props}) => <strong className="text-white font-semibold" {...props} />,
                  em: ({...props}) => <em className="text-gray-300 italic" {...props} />,
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          </div>
        ) : (
          <div className="flex-1 nodrag">
            <ModernEditor
              value={content}
              onChange={handleContentChange}
              language={getEditorLanguage()}
              theme="dark"
              fontSize={fontSize}
              readOnly={false}
              minimap={false}
              lineNumbers={true}
              wordWrap={true}
              autoDetectLanguage={!data?.file_type}
              height="100%"
            />
          </div>
        )}
      </div>
    </div>
  );
} 