import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatView from './ChatView';

describe('ChatView Component', () => {
  const mockOnSubmit = jest.fn();
  const sampleMessages = [
    { sender: 'user', text: 'Hello there!' },
    { sender: 'assistant', text: 'Hi! How can I help you?' },
    { sender: 'user', text: 'Tell me a joke.' },
  ];

  beforeEach(() => {
    // Reset the mock before each test
    mockOnSubmit.mockClear();
  });

  // 1. Rendering Tests
  test('renders without crashing', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    expect(screen.getByPlaceholderText(/type a message.../i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  test('renders input field', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    expect(screen.getByPlaceholderText(/type a message.../i)).toBeInTheDocument();
  });

  test('renders send button', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  // 2. Message Display Tests
  test('displays messages correctly', () => {
    render(<ChatView messages={sampleMessages} onSubmit={mockOnSubmit} />);
    expect(screen.getByText('Hello there!')).toBeInTheDocument();
    expect(screen.getByText('Hi! How can I help you?')).toBeInTheDocument();
    expect(screen.getByText('Tell me a joke.')).toBeInTheDocument();
  });

  test('applies correct Tailwind classes for user and assistant messages', () => {
    render(<ChatView messages={sampleMessages} onSubmit={mockOnSubmit} />);

    const userMessageText = screen.getByText('Hello there!');
    const assistantMessageText = screen.getByText('Hi! How can I help you?');

    // Bubble is the direct parent of the text <p>
    const userMessageBubble = userMessageText.parentElement;
    const assistantMessageBubble = assistantMessageText.parentElement;

    // Message container is the parent of the bubble
    const userMessageContainer = userMessageBubble.parentElement;
    const assistantMessageContainer = assistantMessageBubble.parentElement;

    // User message styling
    expect(userMessageContainer).toHaveClass('self-end');
    expect(userMessageBubble).toHaveClass('bg-blue-500', 'text-white', 'rounded-br-lg');

    // Assistant message styling
    expect(assistantMessageContainer).toHaveClass('self-start');
    expect(assistantMessageBubble).toHaveClass('bg-gray-200', 'text-gray-800', 'rounded-bl-lg');
  });

  test('messagesEndRef div is present for auto-scrolling', () => {
    const { container } = render(<ChatView messages={sampleMessages} onSubmit={mockOnSubmit} />);
    // The messagesEndRef is an empty div at the end of the messages container
    // Its direct parent is the messages container
    const messagesContainer = screen.getByText('Hello there!').closest('.flex-col'); // Heuristic to find messages container
    expect(messagesContainer.lastChild).toBeInTheDocument();
    // Check if it's an empty div, not strictly necessary but good for confirmation
    expect(messagesContainer.lastChild.tagName.toLowerCase()).toBe('div');
    expect(messagesContainer.lastChild.textContent).toBe('');
  });

  // 3. Input Handling Tests
  test('input field value changes on user typing', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);
    fireEvent.change(inputField, { target: { value: 'New message' } });
    expect(inputField.value).toBe('New message');
  });

  test('calls onSubmit with input value when send button is clicked', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);
    const sendButton = screen.getByRole('button', { name: /send/i });

    fireEvent.change(inputField, { target: { value: 'Test submit' } });
    fireEvent.click(sendButton);

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    expect(mockOnSubmit).toHaveBeenCalledWith('Test submit');
    expect(inputField.value).toBe(''); // Input should clear after submit
  });

  test('calls onSubmit when Enter key is pressed in input field', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);

    fireEvent.change(inputField, { target: { value: 'Enter key test' } });
    fireEvent.keyPress(inputField, { key: 'Enter', code: 'Enter', charCode: 13 });

    expect(mockOnSubmit).toHaveBeenCalledTimes(1);
    expect(mockOnSubmit).toHaveBeenCalledWith('Enter key test');
    expect(inputField.value).toBe('');
  });

  test('does not call onSubmit if input is empty and send button is clicked', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const sendButton = screen.getByRole('button', { name: /send/i });
    fireEvent.click(sendButton);
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  test('does not call onSubmit if input is only whitespace and send button is clicked', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);
    const sendButton = screen.getByRole('button', { name: /send/i });

    fireEvent.change(inputField, { target: { value: '   ' } });
    fireEvent.click(sendButton);
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  // 4. Visual Elements (Tailwind CSS Class Checks)
  test('main container has correct Tailwind CSS classes', () => {
    const { container } = render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const mainDiv = container.firstChild;
    expect(mainDiv).toHaveClass('flex', 'flex-col', 'h-full', 'border-gray-300', 'rounded-lg', 'overflow-hidden', 'font-sans', 'bg-white');
  });

  test('messages container has correct Tailwind CSS classes', () => {
    const { container } = render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const mainDiv = container.firstChild;
    const messagesContainer = mainDiv.firstChild; // This is the messages div
    expect(messagesContainer).toHaveClass('flex-grow', 'p-4', 'overflow-y-auto', 'bg-gray-50', 'flex', 'flex-col', 'space-y-3');
  });

  test('chat input area has correct Tailwind CSS classes', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);
    const chatInputDiv = inputField.parentElement;
    expect(chatInputDiv).toHaveClass('flex', 'p-3', 'border-t', 'border-gray-200', 'bg-white', 'items-center');
  });

  test('input field has correct Tailwind CSS classes', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const inputField = screen.getByPlaceholderText(/type a message.../i);
    expect(inputField).toHaveClass('flex-grow', 'py-2', 'px-4', 'border', 'border-gray-300', 'rounded-full', 'mr-3', 'text-sm');
    expect(inputField).toHaveClass('focus:outline-none', 'focus:ring-2', 'focus:ring-blue-500', 'focus:border-transparent');
  });

  test('send button has correct Tailwind CSS classes', () => {
    render(<ChatView messages={[]} onSubmit={mockOnSubmit} />);
    const sendButton = screen.getByRole('button', { name: /send/i });
    expect(sendButton).toHaveClass('py-2', 'px-5', 'bg-blue-500', 'text-white', 'rounded-full', 'cursor-pointer', 'text-sm');
    expect(sendButton).toHaveClass('hover:bg-blue-600', 'focus:outline-none', 'focus:ring-2', 'focus:ring-blue-500', 'focus:ring-opacity-50', 'transition-colors');
  });
});
