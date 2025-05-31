import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { IoSend, IoSettings, IoChatbubbleEllipses, IoTrash, IoAdd, IoCopy, IoRefresh } from 'react-icons/io5';
import ReactMarkdown from 'react-markdown';
import Editor from '../../../../../Input/Editor';
import EntityRenderer from '../EntityRenderer/EntityRenderer';
import useEntityExtractor from '../../../../../../hooks/useEntityExtractor';

interface ChatInterfaceProps {
    data?: ChatInterfaceData;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    viewEntityId: string;
    parentEntityId: string;
}

interface ChatInterfaceData {
    name?: string;
    system_prompt?: string;
    model_type?: string;
    model_name?: string;
    message_history?: Message[];
    current_input?: string;
    auto_scroll?: boolean;
    show_settings?: boolean;
}

interface Message {
    type: 'request' | 'response' | 'context';
    content: string;
    timestamp?: string;
}

export default function ChatInterface({
    data,
    sendStrategyRequest,
    updateEntity,
    viewEntityId,
    parentEntityId,
}: ChatInterfaceProps) {
    const [currentInput, setCurrentInput] = useState(data?.current_input || '');
    const [systemPrompt, setSystemPrompt] = useState(data?.system_prompt || '');
    const [showSettings, setShowSettings] = useState(data?.show_settings || false);
    const [fontSize, setFontSize] = useState(25);
    const [displayMode, setDisplayMode] = useState<'all' | 'request' | 'response'>('all');
    const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    // Get the parent entity directly - it should be the API model
    const parentEntity = useRecoilValue(nodeSelectorFamily(parentEntityId)) as any;

    // Check if the parent entity is an API model
    const currentApiModel = parentEntity?.entity_name === "api_model" ? parentEntity : null;

    // Get messages from the API model's message_history attribute or from view data
    const messages: Message[] = currentApiModel?.data?.message_history || data?.message_history || [];

    const scrollToBottom = useCallback((smooth = false) => {
        if (messagesEndRef.current && data?.auto_scroll !== false) {
            messagesEndRef.current.scrollIntoView({
                behavior: smooth ? 'smooth' : 'auto',
                block: 'end'
            });
        }
    }, [data?.auto_scroll]);

    useEffect(() => {
        const timeoutId = setTimeout(() => {
            scrollToBottom(true);
        }, 100);

        return () => clearTimeout(timeoutId);
    }, [messages, scrollToBottom]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!currentInput.trim() || isSubmitting) return;

        setIsSubmitting(true);

        try {
            // If no API model exists, show error
            if (!currentApiModel) {
                return;
            }

            // Call the model with user input
            sendStrategyRequest(StrategyRequests.builder()
                .withStrategyName('CallApiModelStrategy')
                .withTargetEntity(currentApiModel.entity_id)
                .withParams({
                    user_input: currentInput,
                    system_prompt: systemPrompt,
                    serialize_entities_and_strategies: true
                })
                .withAddToHistory(false)
                .build());

            // Clear input
            setCurrentInput('');

        } catch (error) {
            console.error('Error submitting message:', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    const configureApiModel = () => {
        if (!currentApiModel) return;

        sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('ConfigureApiModelStrategy')
            .withTargetEntity(currentApiModel.entity_id)
            .withParams({
                "env_key": "OPENAI_API_KEY",
                "model_name": "o4-mini-2025-04-16",
                "model_type": "openai",
                "model_config": {
                    "top_p": 1,
                    "stream": false,
                    "max_tokens": 4000,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                }
            })
            .build());
    };

    const clearChatHistory = () => {
        if (!currentApiModel) return;

        sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('ClearChatHistoryStrategy')
            .withTargetEntity(currentApiModel.entity_id)
            .build());
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as any);
        }
    };

    const handleCopy = async (content: string, index: number) => {
        try {
            await navigator.clipboard.writeText(content);
            setCopiedMessageIndex(index);
            setTimeout(() => setCopiedMessageIndex(null), 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    };

    const { extractEntityData } = useEntityExtractor();

    const renderEntityView = (entityData: any, index: number) => {
        return (
            <div className="my-4">
                <EntityRenderer
                    entityData={entityData}
                    sendStrategyRequest={sendStrategyRequest}
                    updateEntity={updateEntity}
                    showBorder={true}
                />
            </div>
        );
    };

    const renderMessageContent = (content: string) => {
        // First check for serialized entity data
        const entityData = extractEntityData(content);
        if (entityData) {
            return (
                <div>
                    {renderEntityView(entityData, 0)}
                    {/* Also render the rest of the content if there's more */}
                    {content.includes('Entity Graph') ? (
                        <div className="mt-4 text-gray-300">
                            <ReactMarkdown>{content.replace(/\s*={50}\r?\n\s*Entity Graph\r?\n\s*-{50}\r?\n\s*\{[\s\S]*?\}\s*\r?\n\s*={50}/, '')}</ReactMarkdown>
                        </div>
                    ) : null}
                </div>
            );
        }

        // Existing code block handling
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

    const filteredMessages = messages.filter(message => {
        if (displayMode === 'all') return true;
        return message.type === displayMode;
    });

    if (!data) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500">
                Loading chat interface...
            </div>
        );
    }

    return (
        <div className="nodrag flex flex-col w-full h-full bg-gray-900 text-white overflow-hidden">
            {/* Header */}
            <div className="flex-shrink-0 p-4 border-b border-gray-700/50">
                <div className="flex justify-between items-center mb-2">
                    <h1 className="text-xl font-bold text-white flex items-center gap-2">
                        {/* @ts-ignore */}
                        <IoChatbubbleEllipses className="text-blue-400" />
                        {data.name || 'AI Chat Interface'}
                    </h1>
                    <div className="flex items-center gap-2">
                        {/* Model Status */}
                        <div className={`px-2 py-1 rounded text-xs ${currentApiModel ? 'bg-green-600' : 'bg-red-600'
                            }`}>
                            {currentApiModel ? 'Model Ready' : 'No API Model'}
                        </div>

                        {/* Action Buttons */}
                        <button
                            onClick={() => setShowSettings(!showSettings)}
                            className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                            title="Settings"
                        >
                            {/* @ts-ignore */}
                            <IoSettings />
                        </button>

                        {currentApiModel && (
                            <>
                                <button
                                    onClick={configureApiModel}
                                    className="p-2 bg-blue-700 rounded hover:bg-blue-600 transition-colors"
                                    title="Configure API Model"
                                >
                                    {/* @ts-ignore */}
                                    <IoRefresh />
                                </button>

                                <button
                                    onClick={clearChatHistory}
                                    className="p-2 bg-red-700 rounded hover:bg-red-600 transition-colors"
                                    title="Clear Chat"
                                >
                                    {/* @ts-ignore */}
                                    <IoTrash />
                                </button>
                            </>
                        )}
                    </div>
                </div>

                {/* Settings Panel */}
                {showSettings && (
                    <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                        <h3 className="text-sm font-semibold mb-3">Chat Settings</h3>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {/* System Prompt */}
                            <div className="md:col-span-2">
                                <label className="block text-xs text-gray-300 mb-1">System Prompt</label>
                                <textarea
                                    value={systemPrompt}
                                    onChange={(e) => setSystemPrompt(e.target.value)}
                                    className="w-full p-2 bg-gray-800 border border-gray-600 rounded text-sm resize-none"
                                    rows={3}
                                    placeholder="Set system instructions for the AI..."
                                />
                            </div>

                            {/* Display Options */}
                            <div className="space-y-3">
                                <div>
                                    <label className="block text-xs text-gray-300 mb-1">Display Mode</label>
                                    <select
                                        value={displayMode}
                                        onChange={(e) => setDisplayMode(e.target.value as any)}
                                        className="w-full p-2 bg-gray-800 border border-gray-600 rounded text-sm"
                                    >
                                        <option value="all">All Messages</option>
                                        <option value="request">User Messages</option>
                                        <option value="response">AI Responses</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-xs text-gray-300 mb-1">Font Size</label>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setFontSize(Math.max(10, fontSize - 1))}
                                            className="px-2 py-1 bg-gray-800 rounded text-xs hover:bg-gray-600"
                                        >
                                            A-
                                        </button>
                                        <span className="text-xs min-w-[3rem] text-center">{fontSize}px</span>
                                        <button
                                            onClick={() => setFontSize(Math.min(20, fontSize + 1))}
                                            className="px-2 py-1 bg-gray-800 rounded text-xs hover:bg-gray-600"
                                        >
                                            A+
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Messages */}
            <div
                className="nowheel flex-grow min-h-0 overflow-y-auto p-4"
                ref={chatContainerRef}
                style={{ fontSize: `${fontSize}px` }}
            >
                {filteredMessages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                        {/* @ts-ignore */}
                        <IoChatbubbleEllipses className="text-6xl mb-4 nodrag" />
                        <p className="text-lg mb-2">Start a conversation</p>
                        <p className="text-sm">
                            {currentApiModel
                                ? "Type a message below to begin chatting with the AI"
                                : "Parent entity must be an API model to use this chat interface"
                            }
                        </p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {filteredMessages.map((message, index) => (
                            <div
                                key={index}
                                className="w-full group"
                            >
                                <div
                                    className={`relative w-full max-w-none p-6 ${
                                        message.type === 'response'
                                            ? 'bg-gray-800/30 border-l-2 border-gray-600/30'
                                            : message.type === 'context'
                                                ? 'bg-blue-900/20 border-l-2 border-blue-500/40'
                                                : 'bg-gray-700/20 border-l-2 border-gray-500/30'
                                    }`}
                                >
                                    {/* Copy Button */}
                                    <button
                                        onClick={() => handleCopy(message.content, index)}
                                        className="absolute top-3 right-3 p-2 rounded-md hover:bg-gray-600/30 transition-colors opacity-0 group-hover:opacity-100"
                                        title="Copy to clipboard"
                                    >
                                        {copiedMessageIndex === index ? (
                                            <span className="text-green-400 text-sm">✓</span>
                                        ) : (
                                            /* @ts-ignore */
                                            <IoCopy className="text-gray-400 text-sm" />
                                        )}
                                    </button>

                                    {/* Message Type Badge */}
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className={`text-sm font-semibold ${
                                            message.type === 'response' 
                                                ? 'text-blue-400' 
                                                : message.type === 'context'
                                                    ? 'text-purple-400'
                                                    : 'text-green-400'
                                        }`}>
                                            {message.type === 'response' ? 'Assistant' : 
                                             message.type === 'context' ? 'Context' : 'You'}
                                        </span>
                                        {message.timestamp && (
                                            <span className="text-xs text-gray-500">
                                                {new Date(message.timestamp).toLocaleTimeString()}
                                            </span>
                                        )}
                                    </div>

                                    {/* Message Content */}
                                    <div className="pr-12 text-gray-100 leading-relaxed">
                                        {renderMessageContent(message.content)}
                                    </div>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="flex-shrink-0 p-6 border-t border-gray-700/50 bg-gray-800/30">
                <form onSubmit={handleSubmit} className="space-y-3">
                    <div className="flex gap-3">
                        <div className="flex-grow relative">
                            <textarea
                                ref={inputRef}
                                value={currentInput}
                                onChange={(e) => setCurrentInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder={
                                    currentApiModel
                                        ? "Type your message... (Shift+Enter for new line)"
                                        : "Parent entity must be an API model..."
                                }
                                disabled={!currentApiModel || isSubmitting}
                                className="w-full p-4 pr-14 bg-gray-700/50 border border-gray-600/50 rounded-xl resize-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 focus:outline-none disabled:opacity-50 text-gray-100 placeholder-gray-400"
                                rows={2}
                                style={{ minHeight: '3rem' }}
                            />

                            <button
                                type="submit"
                                disabled={!currentInput.trim() || !currentApiModel || isSubmitting}
                                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-2.5 bg-blue-600/80 rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                            >
                                {isSubmitting ? (
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                ) : (
                                    /* @ts-ignore */
                                    <IoSend className="w-4 h-4 text-white" />
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Status Bar */}
                    <div className="flex justify-between items-center text-xs text-gray-500">
                        <div>
                            {messages.length > 0 && `${messages.length} messages`}
                            {currentApiModel && ` • Model: ${currentApiModel.data?.model_name || 'Unknown'}`}
                        </div>
                        <div>
                            {isSubmitting && 'Sending...'}
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
} 