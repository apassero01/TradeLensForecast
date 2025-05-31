import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { IoSave, IoEye, IoCode, IoDocumentText } from 'react-icons/io5';
import ReactMarkdown from 'react-markdown';
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
  const [showPreview, setShowPreview] = useState(false);
  const [fontSize, setFontSize] = useState(14);
  const editorRef = useRef<any>(null);

  // Update content when data changes
  useEffect(() => {
    if (data?.text !== undefined && data.text !== content) {
      setContent(data.text);
      setIsDirty(false);
    }
  }, [data?.text]);

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

  const isMarkdown = data?.file_type === 'markdown' || data?.file_type === 'md';

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
              onClick={() => setShowPreview(!showPreview)}
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
      <div className="flex-1 overflow-hidden" onKeyDown={handleKeyDown}>
        {showPreview && isMarkdown ? (
          <div className="h-full overflow-auto p-6 bg-gray-800">
            <div className="prose prose-invert max-w-none" style={{ fontSize: `${fontSize}px` }}>
              <ReactMarkdown>{content}</ReactMarkdown>
            </div>
          </div>
        ) : (
          <div className="h-full nodrag">
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