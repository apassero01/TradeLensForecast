import React, { useState, useEffect, useRef } from 'react';
import Editor from '../../../../Input/Editor';
import ReactMarkdown from 'react-markdown';
import 'ace-builds/src-noconflict/theme-monokai';

const ChatScreen = ({ visualization }) => {
  const [fontSize, setFontSize] = useState(14);
  const [displayMode, setDisplayMode] = useState('all');
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);

  const hasData = !!(visualization && visualization.data && visualization.data.response);
  const messages = hasData ? visualization.data.response : [];

  const scrollToBottom = (smooth = false) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: smooth ? 'smooth' : 'auto',
        block: 'end'
      });
    }
  };

  // Scroll to bottom when messages change or component mounts
  useEffect(() => {
    // Use a small timeout to ensure DOM is fully updated
    const timeoutId = setTimeout(() => {
      scrollToBottom();
    }, 50);
    
    return () => clearTimeout(timeoutId);
  }, [messages, displayMode, visualization?.data]);

  // Also attach a mutation observer to watch for dynamic content changes
  useEffect(() => {
    if (!chatContainerRef.current) return;
    
    const observer = new MutationObserver(() => {
      scrollToBottom();
    });
    
    observer.observe(chatContainerRef.current, {
      childList: true,
      subtree: true
    });
    
    return () => observer.disconnect();
  }, []);

  if (!hasData) {
    return <div className="text-red-500">No chat messages available</div>;
  }

  const handleFontSize = (change) => {
    setFontSize((prev) => Math.max(8, Math.min(24, prev + change)));
  };

  const handleModeChange = (e) => {
    setDisplayMode(e.target.value);
  };

  const filteredMessages = messages.filter(message => {
    if (displayMode === 'all') return true;
    return message.type === displayMode;
  });

  const renderMessageContent = (content) => {
    const codeBlockRegex = /```([\w]*)\n?([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push({
          type: 'markdown',
          content: content.slice(lastIndex, match.index)
        });
      }

      const language = match[1].trim() || 'text';
      const code = match[2].trim();
      parts.push({
        type: 'code',
        language,
        content: code
      });

      lastIndex = match.index + match[0].length;
    }

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
        // Determine editor type and title based on language

        const editorType = part.language === 'StrategyRequest' ? 'json' : part.language;
        const editorTitle = part.language === 'StrategyRequest' 
          ? 'Strategy Request' 
          : `${part.language.toUpperCase()} Code Block`;

        const editorVisualization = {
          data: part.content,
          config: {
            type: editorType,
            title: editorTitle,
            readOnly: true
          }
        };

        const numberOfLines = part.content.split('\n').length;
        const lineHeight = 20;
        const headerHeight = 56;
        const minHeight = 100;
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

  const handleCopy = async (content, messageId) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const handleWheel = (e) => {
    e.stopPropagation();
  };

  return (
    <div className="flex flex-col w-full h-full nodrag bg-gray-800">
      <div className="flex-none border-b border-gray-700 p-2 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <h2 className="text-gray-200 text-lg">
            {visualization.config?.title || 'Chat History'}
          </h2>
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

      <div 
        className="flex-grow min-h-0 overflow-y-auto p-4 nowheel" 
        onWheel={handleWheel}
        ref={chatContainerRef}
      >
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
          <div ref={messagesEndRef} className="h-[1px]" />
        </div>
      </div>
    </div>
  );
};

export default ChatScreen;