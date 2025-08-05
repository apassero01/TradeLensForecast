import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../../../state/entitiesSelectors';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { IoSend, IoClose, IoExpand, IoContract, IoSwapHorizontal, IoSearch } from 'react-icons/io5';
import EntityViewRenderer from '../ChatInterface/EntityViewRenderer';

interface EntityCentricChatViewProps {
    data?: EntityCentricChatViewData;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    viewEntityId: string;
    parentEntityId: string;
}

interface EntityCentricChatViewData {
    name?: string;
    visible_entities?: string[];
    sort_by?: 'recent' | 'child_count' | 'name';
    sort_order?: 'asc' | 'desc';
    auto_scroll?: boolean;
    show_search?: boolean;
}


export default function EntityCentricChatView({
    data,
    sendStrategyRequest,
    updateEntity,
    viewEntityId,
    parentEntityId,
}: EntityCentricChatViewProps) {
    const [currentInput, setCurrentInput] = useState('');
    const [fullscreenEntity, setFullscreenEntity] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [showSearch, setShowSearch] = useState(data?.show_search || false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const entityListRef = useRef<HTMLDivElement>(null);

    // Get the parent entity (API model)
    const parentEntity = useRecoilValue(nodeSelectorFamily(parentEntityId)) as any;
    const currentApiModel = parentEntity?.entity_name === "api_model" ? parentEntity : null;

    // Get visible entities from API model's visible_entities attribute
    const visibleEntityIds = currentApiModel?.data?.visible_entities || data?.visible_entities || [];

    // Filter entities based on search - for now just filter IDs
    // TODO: Implement proper search with entity names
    const filteredEntityIds = visibleEntityIds;

    // Get the most recent message from the API model
    const messages = currentApiModel?.data?.message_history || [];
    const lastMessage = messages[messages.length - 1];

    // Handle entity selection
    const handleEntityClick = (entityId: string) => {
        setFullscreenEntity(entityId);
    };

    // Handle closing entity from view
    const handleCloseEntity = (entityId: string, e?: React.MouseEvent) => {
        e?.stopPropagation();
        
        if (!currentApiModel) return;

        // Remove from visible_entities
        const updatedVisibleEntities = visibleEntityIds.filter((id: string) => id !== entityId);
        
        sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('SetAttributesStrategy')
            .withTargetEntity(currentApiModel.entity_id)
            .withParams({
                attributes: {
                    visible_entities: updatedVisibleEntities
                }
            })
            .withAddToHistory(false)
            .build());
    };

    // Handle tab navigation between entities
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Cmd+Tab or Ctrl+Tab to switch between entities
            if ((e.metaKey || e.ctrlKey) && e.key === 'Tab') {
                e.preventDefault();
                
                if (filteredEntityIds.length === 0) return;
                
                const currentIndex = fullscreenEntity 
                    ? filteredEntityIds.indexOf(fullscreenEntity)
                    : -1;
                    
                const nextIndex = e.shiftKey 
                    ? (currentIndex - 1 + filteredEntityIds.length) % filteredEntityIds.length
                    : (currentIndex + 1) % filteredEntityIds.length;
                    
                setFullscreenEntity(filteredEntityIds[nextIndex]);
            }
            
            // Escape to close fullscreen
            if (e.key === 'Escape' && fullscreenEntity) {
                setFullscreenEntity(null);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [fullscreenEntity, filteredEntityIds]);

    // Handle message submission
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!currentInput.trim() || isSubmitting || !currentApiModel) return;

        setIsSubmitting(true);

        try {
            sendStrategyRequest(StrategyRequests.builder()
                .withStrategyName('CallApiModelStrategy')
                .withTargetEntity(currentApiModel.entity_id)
                .withParams({
                    user_input: currentInput,
                    serialize_entities_and_strategies: true
                })
                .withAddToHistory(false)
                .build());

            setCurrentInput('');
        } catch (error) {
            console.error('Error submitting message:', error);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as any);
        }
    };

    // Manual entity search
    const handleEntitySearch = () => {
        if (!searchQuery.trim()) return;
        
        sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('SearchEntitiesStrategy')
            .withTargetEntity(parentEntityId)
            .withParams({
                query: searchQuery,
                limit: 10
            })
            .build());
    };

    return (
        <div className="nodrag flex flex-col w-full h-full bg-gray-900 text-white overflow-hidden">
            {/* Entity Visualization Area - Primary Focus */}
            <div className="flex-grow flex flex-col min-h-0">
                {/* Entity List Header */}
                <div className="flex-shrink-0 p-3 border-b border-gray-700/50 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-gray-300">
                        Entities ({filteredEntityIds.length})
                    </h2>
                    <div className="flex items-center gap-2">
                        {/* Sort Options */}
                        <select
                            value={`${data?.sort_by || 'recent'}-${data?.sort_order || 'desc'}`}
                            onChange={(e) => {
                                const [sortBy, sortOrder] = e.target.value.split('-');
                                updateEntity(viewEntityId, {
                                    ...data,
                                    sort_by: sortBy,
                                    sort_order: sortOrder
                                });
                            }}
                            className="text-xs bg-gray-800 border border-gray-700 rounded px-2 py-1"
                        >
                            <option value="recent-desc">Most Recent</option>
                            <option value="recent-asc">Least Recent</option>
                            <option value="child_count-asc">Fewest Children</option>
                            <option value="child_count-desc">Most Children</option>
                            <option value="name-asc">Name (A-Z)</option>
                            <option value="name-desc">Name (Z-A)</option>
                        </select>
                        
                        {/* Search Toggle */}
                        <button
                            onClick={() => setShowSearch(!showSearch)}
                            className="p-1.5 bg-gray-800 rounded hover:bg-gray-700 transition-colors"
                            title="Search entities"
                        >
                            {/* @ts-ignore */}
                            <IoSearch className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                {/* Search Bar */}
                {showSearch && (
                    <div className="flex-shrink-0 p-3 border-b border-gray-700/50">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleEntitySearch()}
                                placeholder="Search entities..."
                                className="flex-grow px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-sm"
                            />
                            <button
                                onClick={handleEntitySearch}
                                className="px-3 py-1.5 bg-blue-600 rounded hover:bg-blue-500 text-sm"
                            >
                                Search
                            </button>
                        </div>
                    </div>
                )}

                {/* Horizontally Scrollable Entity List */}
                <div 
                    ref={entityListRef}
                    className="flex-grow overflow-x-auto overflow-y-hidden p-4"
                >
                    <div className="flex gap-4 h-full">
                        {filteredEntityIds.length === 0 ? (
                            <div className="flex items-center justify-center w-full text-gray-500">
                                <p className="text-center">
                                    No entities to display.<br/>
                                    Entities will appear here when the AI model interacts with them.
                                </p>
                            </div>
                        ) : (
                            filteredEntityIds.map((entityId: string) => (
                                <div
                                    key={entityId}
                                    onClick={() => handleEntityClick(entityId)}
                                    className="flex-shrink-0 w-48 h-48 bg-gray-800 rounded-lg border border-gray-700 hover:border-blue-500 transition-all cursor-pointer relative group"
                                >
                                    {/* Close Button */}
                                    <button
                                        onClick={(e) => handleCloseEntity(entityId, e)}
                                        className="absolute top-2 right-2 p-1 bg-red-600 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10"
                                        title="Remove from view"
                                    >
                                        {/* @ts-ignore */}
                                        <IoClose className="w-3 h-3" />
                                    </button>
                                    
                                    {/* Entity Preview */}
                                    <div className="h-full w-full overflow-hidden rounded-lg">
                                        <EntityViewRenderer
                                            entityId={entityId}
                                            sendStrategyRequest={sendStrategyRequest}
                                            updateEntity={updateEntity}
                                        />
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Minimized Chat Area - Secondary */}
            <div className="flex-shrink-0 border-t border-gray-700/50 bg-gray-800/30">
                {/* Last Message Display */}
                {lastMessage && (
                    <div className="p-3 border-b border-gray-700/50">
                        <div className="text-xs text-gray-500 mb-1">
                            Last {lastMessage.type === 'human' ? 'User' : 'Assistant'} Message:
                        </div>
                        <div className="text-sm text-gray-300 line-clamp-2">
                            {lastMessage.content}
                        </div>
                    </div>
                )}

                {/* Input Area */}
                <div className="p-4">
                    <form onSubmit={handleSubmit} className="flex gap-2">
                        <textarea
                            ref={inputRef}
                            value={currentInput}
                            onChange={(e) => setCurrentInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={
                                currentApiModel
                                    ? "Type your message..."
                                    : "Parent must be an API model..."
                            }
                            disabled={!currentApiModel || isSubmitting}
                            className="flex-grow px-3 py-2 bg-gray-700/50 border border-gray-600/50 rounded-lg resize-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 focus:outline-none disabled:opacity-50 text-sm"
                            rows={1}
                        />
                        <button
                            type="submit"
                            disabled={!currentInput.trim() || !currentApiModel || isSubmitting}
                            className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            {isSubmitting ? (
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            ) : (
                                /* @ts-ignore */
                                <IoSend className="w-4 h-4" />
                            )}
                        </button>
                    </form>
                </div>
            </div>

            {/* Fullscreen Entity View */}
            {fullscreenEntity && (
                <div className="fixed inset-0 bg-gray-900 z-50 flex flex-col">
                    {/* Fullscreen Content - No Header */}
                    <div className="flex-grow overflow-auto">
                        <EntityViewRenderer
                            entityId={fullscreenEntity}
                            sendStrategyRequest={sendStrategyRequest}
                            updateEntity={updateEntity}
                        />
                    </div>
                    
                    {/* Navigation Hint */}
                    <div className="flex-shrink-0 p-2 bg-gray-800 text-center text-xs text-gray-500">
                        Use Cmd+Tab to switch between entities • Esc to close
                    </div>
                </div>
            )}
        </div>
    );
}