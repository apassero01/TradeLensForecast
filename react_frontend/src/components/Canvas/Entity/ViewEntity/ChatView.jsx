import React, { useState, useRef, useEffect } from 'react';

const ChatView = ({ messages, onSubmit }) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleSubmit = () => {
    if (inputValue.trim() !== '') {
      onSubmit(inputValue);
      setInputValue('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent newline on Enter
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full border border-gray-300 rounded-lg overflow-hidden font-sans bg-white">
      <div className="flex-grow p-4 overflow-y-auto bg-gray-50 flex flex-col space-y-3">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex max-w-xs md:max-w-md lg:max-w-lg ${
              msg.sender === 'user' ? 'self-end' : 'self-start'
            }`}
          >
            <div
              className={`py-2 px-4 rounded-2xl shadow ${
                msg.sender === 'user'
                  ? 'bg-blue-500 text-white rounded-br-lg'
                  : 'bg-gray-200 text-gray-800 rounded-bl-lg'
              }`}
            >
              <p className="m-0 text-sm whitespace-pre-wrap">{msg.text}</p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="flex p-3 border-t border-gray-200 bg-white items-center">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Type a message..."
          className="flex-grow py-2 px-4 border border-gray-300 rounded-full mr-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          onClick={handleSubmit}
          className="py-2 px-5 bg-blue-500 text-white rounded-full cursor-pointer text-sm hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatView;
