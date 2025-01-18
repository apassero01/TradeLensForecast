import React, { useState, useEffect, useRef } from 'react';
import Editor from './Editor';
import ReactMarkdown from 'react-markdown';
import 'ace-builds/src-noconflict/theme-monokai';

const ChatScreen = ({ visualization }) => {
  // 1) Always call Hooks at the top
  const [fontSize, setFontSize] = useState(14);
  const [displayMode, setDisplayMode] = useState('all');
  const messagesEndRef = useRef(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);

  // 2) Check if we have data
  const hasData = !!(visualization && visualization.data);
  const messages = hasData ? visualization.data : [];

  // Scroll to bottom whenever messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'auto' });
    }
  }, [messages, displayMode]); // Re-scroll when messages or display mode changes

  // 3) If there's no data, show a message
  if (!hasData) {
    return <div className="text-red-500">No chat messages available</div>;
  }

  // 4) Handle font size changes
  const handleFontSize = (change) => {
    setFontSize((prev) => Math.max(8, Math.min(24, prev + change)));
  };

  // 5) Handle display mode changes
  const handleModeChange = (e) => {
    setDisplayMode(e.target.value);
  };

  // 6) Filter messages based on display mode
  const filteredMessages = messages.filter(message => {
    if (displayMode === 'all') return true;
    return message.type === displayMode;
  });

  // Parse message content for code blocks
  const renderMessageContent = (content) => {
    const codeBlockRegex = /```([\w]*)\n?([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block as markdown
      if (match.index > lastIndex) {
        parts.push({
          type: 'markdown',
          content: content.slice(lastIndex, match.index)
        });
      }

      // Add code block
      const language = match[1].trim() || 'text';
      const code = match[2].trim();
      parts.push({
        type: 'code',
        language,
        content: code
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text as markdown
    if (lastIndex < content.length) {
      parts.push({
        type: 'markdown',
        content: content.slice(lastIndex)
      });
    }

    return parts.map((part, index) => {
      if (part.type === 'markdown') {
        return (
          <div key={index} className="prose prose-invert max-w-none">
            <ReactMarkdown>{part.content}</ReactMarkdown>
          </div>
        );
      } else {
        const editorVisualization = {
          data: part.content,
          config: {
            type: part.language,
            title: `${part.language.toUpperCase()} Code Block`,
            readOnly: true
          }
        };
        
        // Calculate approximate height based on number of lines
        const numberOfLines = part.content.split('\n').length;
        const lineHeight = 20; // Approximate height per line in pixels
        const headerHeight = 56; // Height of the editor header
        const minHeight = 100; // Minimum height
        const height = Math.max(minHeight, (numberOfLines * lineHeight) + headerHeight);
        
        return (
          <div 
            key={index} 
            className="my-4 border border-gray-700 rounded-lg overflow-hidden" 
            style={{ height }}
          >
            <Editor visualization={editorVisualization} />
          </div>
        );
      }
    });
  };

  // Add copy functionality
  const handleCopy = async (content, messageId) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  return (
    <div className="flex flex-col w-full h-full nodrag bg-gray-800">
      <div className="flex-none border-b border-gray-700 p-2 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <h2 className="text-gray-200 text-lg">
            {visualization.config?.title || 'Chat History'}
          </h2>
          {/* Dropdown for filtering message types */}
          <select
            className="bg-gray-700 text-gray-200 p-1 rounded"
            value={displayMode}
            onChange={handleModeChange}
          >
            <option value="all">All Messages</option>
            <option value="response">Responses</option>
            <option value="context">Context</option>
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

      <div className="flex-grow min-h-0 overflow-y-auto p-4">
        <div className="flex flex-col space-y-4" style={{ fontSize: `${fontSize}px` }}>
          {filteredMessages.map((message, index) => (
            <div 
              key={index} 
              className={`flex ${message.type === 'response' ? 'justify-start' : 'justify-end'}`}
            >
              <div 
                className={`relative max-w-[80%] p-4 rounded-lg ${
                  message.type === 'response' 
                    ? 'bg-gray-700 text-white' 
                    : 'bg-blue-600 text-white'
                }`}
              >
                <button
                  onClick={() => handleCopy(message.content, index)}
                  className="absolute top-2 right-2 p-1 rounded hover:bg-gray-600/50 transition-colors"
                  title="Copy to clipboard"
                >
                  {copiedMessageId === index ? (
                    <span className="text-green-400 text-sm">✓</span>
                  ) : (
                    <span className="text-gray-300 text-sm">📋</span>
                  )}
                </button>
                <div className="text-sm text-gray-300 mb-1">
                  {message.type.toUpperCase()}
                </div>
                {renderMessageContent(message.content)}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  );
};

export default ChatScreen;