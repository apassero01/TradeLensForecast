import React, { memo } from 'react';
import { IoCopy, IoDocumentText } from 'react-icons/io5';
import ReactMarkdown from 'react-markdown';
import Editor from '../../../../../Input/Editor';
import EntityViewRenderer from './EntityViewRenderer';

interface Message {
    type: 'ai' | 'human' | 'system' | 'tool';
    content: string;
    timestamp?: string;
}

interface MessageItemProps {
    message: Message;
    index: number;
    fontSize: number;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    currentApiModel: any;
    parentEntityId: string;
    onCopy: (content: string, index: number) => void;
    onCreateDocument: (content: string, index: number, messageType: string) => void;
    copiedMessageIndex: number | null;
    createdDocumentIndex: number | null;
    setModalEntity: (entityId: string | null) => void;
}

const MessageItem = memo(({ 
    message, 
    index, 
    fontSize, 
    sendStrategyRequest, 
    updateEntity, 
    currentApiModel, 
    parentEntityId,
    onCopy,
    onCreateDocument,
    copiedMessageIndex,
    createdDocumentIndex,
    setModalEntity
}: MessageItemProps) => {
    const renderMessageContent = (content: string, messageType: string) => {
        // First extract entity IDs from ```entities``` tags
        let entityIds: string[] = [];
        const entityTagRegex = /```entities\n([\s\S]*?)```/g;
        let contentWithoutEntityTags = content;
        let entityMatch;
        
        while ((entityMatch = entityTagRegex.exec(content)) !== null) {
            try {
                const ids = JSON.parse(entityMatch[1]);
                if (Array.isArray(ids)) {
                    entityIds = entityIds.concat(ids);
                }
                // Remove the entity tag from content
                contentWithoutEntityTags = contentWithoutEntityTags.replace(entityMatch[0], '');
            } catch (e) {
                console.error('Failed to parse entity IDs:', e);
            }
        }
        
        // Remove duplicates
        entityIds = [...new Set(entityIds)];
        
        // For tool messages, if we have entity IDs, just show the icons
        if (messageType === 'tool' && entityIds.length > 0) {
            return (
                <div className="flex items-center gap-2">
                    <div className="text-xs text-gray-400">Affected entities:</div>
                    <div className="flex gap-2">
                        {entityIds.map((entityId, index) => (
                            <EntityViewRenderer
                                key={`${entityId}-${index}`}
                                entityId={entityId}
                                sendStrategyRequest={sendStrategyRequest}
                                updateEntity={updateEntity}
                                isIcon={true}
                                onIconClick={() => setModalEntity(entityId)}
                            />
                        ))}
                    </div>
                </div>
            );
        }
        
        // Parse remaining content for code blocks
        const codeBlockRegex = /```([\w]*)\n?([\s\S]*?)```/g;
        const parts = [];
        let lastIndex = 0;
        let match;

        while ((match = codeBlockRegex.exec(contentWithoutEntityTags)) !== null) {
            if (match.index > lastIndex) {
                parts.push({
                    type: 'markdown',
                    content: contentWithoutEntityTags.slice(lastIndex, match.index)
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

        if (lastIndex < contentWithoutEntityTags.length) {
            parts.push({
                type: 'markdown',
                content: contentWithoutEntityTags.slice(lastIndex)
            });
        }

        return (
            <>
                {/* Render message content first */}
                {parts.map((part, partIndex) => {
                    if (part.type === 'markdown') {
                        return (
                            <div key={partIndex} className="prose prose-invert max-w-none">
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
                                key={partIndex}
                                className="my-4 border border-gray-700 rounded-lg overflow-hidden"
                                style={{ height }}
                            >
                                <Editor visualization={editorVisualization} />
                            </div>
                        );
                    }
                })}
                
                {/* Render entity views if any (for non-tool messages) */}
                {messageType !== 'tool' && entityIds.length > 0 && (
                    <div className="mt-4 space-y-3">
                        <div className="text-xs text-gray-400 uppercase tracking-wider">Affected Entities:</div>
                        {entityIds.map((entityId, entityIndex) => (
                            <EntityViewRenderer
                                key={`${entityId}-${entityIndex}`}
                                entityId={entityId}
                                sendStrategyRequest={sendStrategyRequest}
                                updateEntity={updateEntity}
                            />
                        ))}
                    </div>
                )}
            </>
        );
    };

    return (
        <div className="w-full group mb-6" style={{ fontSize: `${fontSize}px` }}>
            <div
                className={`relative w-full max-w-none p-6 ${
                    message.type === 'ai'
                        ? 'bg-gray-800/30 border-l-2 border-gray-600/30'
                        : message.type === 'system' || message.type === 'tool'
                            ? 'bg-blue-900/20 border-l-2 border-blue-500/40'
                            : 'bg-gray-700/20 border-l-2 border-gray-500/30'
                }`}
            >
                {/* Copy Button */}
                <button
                    onClick={() => onCopy(message.content, index)}
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

                {/* Create Document Button */}
                <button
                    onClick={() => onCreateDocument(message.content, index, message.type)}
                    className="absolute top-3 right-12 p-2 rounded-md hover:bg-gray-600/30 transition-colors opacity-0 group-hover:opacity-100"
                    title="Create document from message"
                >
                    {createdDocumentIndex === index ? (
                        <span className="text-green-400 text-sm">✓</span>
                    ) : (
                        /* @ts-ignore */
                        <IoDocumentText className="text-gray-400 text-sm" />
                    )}
                </button>

                {/* Message Type Badge */}
                <div className="flex items-center gap-3 mb-4">
                    <span className={`text-sm font-semibold ${
                        message.type === 'ai' 
                            ? 'text-blue-400' 
                            : message.type === 'system'
                                ? 'text-purple-400'
                                : 'text-green-400'
                    }`}>
                        {message.type === 'ai' ? 'Assistant' : 
                         message.type === 'tool' ? 'Tool' : 'You'}
                    </span>
                    {message.timestamp && (
                        <span className="text-xs text-gray-500">
                            {new Date(message.timestamp).toLocaleTimeString()}
                        </span>
                    )}
                </div>

                {/* Message Content */}
                <div className="pr-12 text-gray-100 leading-relaxed">
                    {renderMessageContent(message.content, message.type)}
                </div>
            </div>
        </div>
    );
});

MessageItem.displayName = 'MessageItem';

export default MessageItem;