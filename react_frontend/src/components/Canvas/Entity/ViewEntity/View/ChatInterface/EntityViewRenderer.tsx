import React, { useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import viewComponents from '../viewComponents';
import { FaArrowLeft } from 'react-icons/fa';
import { useRecoilState } from 'recoil';
import { sessionAtom } from '../../../../../../state/sessionAtoms';


interface EntityViewRendererProps {
    entityId: string;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    isIcon?: boolean;
    onIconClick?: () => void;
    initialViewId?: string;
    isStandalone?: boolean; // New prop to indicate if this is a standalone page view
}

export default function EntityViewRenderer({ 
    entityId, 
    sendStrategyRequest, 
    updateEntity,
    isIcon = false,
    onIconClick,
    initialViewId,
    isStandalone = false // New prop
}: EntityViewRendererProps) {
    const [, setSession] = useRecoilState(sessionAtom);
    
    const handleBackToCanvas = () => {
        setSession(prev => ({
            ...prev,
            viewMode: 'canvas',
            currentEntityId: null
        }));
    };
    
    // Get all view children
    const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: entityId, type: EntityTypes.VIEW })) as any[];
    
    // Track the currently selected view
    const [selectedViewId, setSelectedViewId] = useState<string | null>(
        initialViewId || (viewChildren.length > 0 ? viewChildren[0].data.entity_id : null)
    );
    
    // State for creating new views
    const [showCreateView, setShowCreateView] = useState(false);
    const [selectedViewType, setSelectedViewType] = useState('');

    // Get the selected view content
    const selectedView = useRenderStoredView(selectedViewId, sendStrategyRequest, updateEntity, {});
    
    // View component names mapping
    const viewComponentNames: Record<string, string> = {
        histogram: 'Histogram',
        linegraph: 'Line Graph',
        stockchart: 'Stock Chart',
        multiline: 'Multi Line',
        editor: 'Editor',
        chatinterface: 'Chat Interface',
        photo: 'Photo Display',
        recipeinstructions: 'Recipe Instructions',
        recipelistitem: 'Recipe List Item',
        recipelist: 'Recipe List',
        newline: 'New Line',
        document_list_item: 'Document List Item',
        mealplan: 'Meal Plan',
        mealplannerdashboard: 'Meal Planner Dashboard',
        entityrenderer: 'Entity Renderer',
        ide_app_dashboard: 'IDE App Dashboard',
        file_tree: 'File Tree',
        document_search: 'Document Search',
        advanced_document_editor: 'Advanced Document Editor',
        calendar_event_details: 'Calendar Event Details',
        calendar_monthly_view: 'Calendar Monthly View',
        entity_centric_chat_view: 'Entity Centric Chat View',
    };

    // Handle creating a new view
    const handleCreateView = () => {
        if (!selectedViewType) {
            alert('Please select a view type');
            return;
        }

        const createRequest = StrategyRequests.createEntity(
            entityId,
            "shared_utils.entities.view_entity.ViewEntity.ViewEntity",
            {
                hidden: false,
                width: 350,
                height: 350,
                view_component_type: selectedViewType
            }
        );

        sendStrategyRequest(createRequest);
        setShowCreateView(false);
        setSelectedViewType('');
    };

    // Handle icon mode separately
    if (isIcon) {
        const content = selectedView || (
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Entity ID:</div>
                <div className="text-sm font-mono text-gray-300">{entityId}</div>
            </div>
        );

        return (
            <div 
                className="w-12 h-12 overflow-hidden cursor-pointer hover:scale-110 transition-transform border border-gray-600 rounded-lg"
                onClick={onIconClick}
                title={`Click to view entity: ${entityId}`}
            >
                <div className="scale-[0.2] origin-top-left w-[240px] h-[240px]">
                    {content}
                </div>
            </div>
        );
    }

    // No views available
    if (viewChildren.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center h-full p-8">
                <div className="text-center space-y-4 max-w-md">
                    <div className="text-6xl text-gray-600 mx-auto">⚙️</div>
                    <h2 className="text-xl font-semibold text-gray-300">No Views Available</h2>
                    <p className="text-sm text-gray-400">
                        This entity doesn't have any views yet. Create one to get started.
                    </p>
                    <div className="text-xs text-gray-500">Entity ID: {entityId}</div>
                    
                    {showCreateView ? (
                        <div className="space-y-3 p-4 bg-gray-800 rounded-lg border border-gray-700">
                            <select
                                className="w-full p-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:border-blue-500"
                                value={selectedViewType}
                                onChange={(e) => setSelectedViewType(e.target.value)}
                            >
                                <option value="">Select a view type...</option>
                                {Object.entries(viewComponents).map(([key]) => (
                                    <option key={key} value={key}>
                                        {viewComponentNames[key] || key}
                                    </option>
                                ))}
                            </select>
                            <div className="flex gap-2">
                                <button
                                    onClick={handleCreateView}
                                    className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
                                >
                                    Create
                                </button>
                                <button
                                    onClick={() => {
                                        setShowCreateView(false);
                                        setSelectedViewType('');
                                    }}
                                    className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-md transition-colors"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    ) : (
                        <button
                            onClick={() => setShowCreateView(true)}
                            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors inline-flex items-center gap-2"
                        >
                            <span>➕</span>
                            Create Your First View
                        </button>
                    )}
                </div>
            </div>
        );
    }

    // Always show tabs (even for single view) to include back button
    return (
        <div className="flex flex-col h-full">
            {/* View selector tabs */}
            <div className="flex items-center gap-2 p-2 bg-gray-800 border-b border-gray-700">
                {/* Back to Canvas button - always show */}
                <button
                    onClick={handleBackToCanvas}
                    className="flex items-center gap-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md transition-colors text-sm"
                    title="Back to canvas"
                >
                    {/* @ts-ignore */}
                    <FaArrowLeft className="text-xs" />
                    <span>Canvas</span>
                </button>
                
                <div className="flex gap-2 flex-1 overflow-x-auto">
                    {viewChildren.map((view) => {
                        const viewName = view.data.view_component_type 
                            ? viewComponentNames[view.data.view_component_type] || view.data.view_component_type
                            : `View ${view.data.entity_id.substring(0, 8)}`;
                        const isSelected = view.data.entity_id === selectedViewId;
                        
                        return (
                            <button
                                key={view.data.entity_id}
                                onClick={() => setSelectedViewId(view.data.entity_id)}
                                className={`px-3 py-1 rounded-md text-sm transition-colors whitespace-nowrap ${
                                    isSelected 
                                        ? 'bg-blue-600 text-white' 
                                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                }`}
                            >
                                {viewName}
                            </button>
                        );
                    })}
                </div>
                
                {/* Add new view button */}
                <button
                    onClick={() => setShowCreateView(!showCreateView)}
                    className="p-1.5 bg-gray-700 hover:bg-gray-600 rounded-md transition-colors flex items-center gap-1 text-sm"
                    title="Add new view"
                >
                    <span className="text-xs">➕</span>
                </button>
            </div>
            
            {/* Create view dropdown */}
            {showCreateView && (
                <div className="p-3 bg-gray-750 border-b border-gray-700">
                    <div className="flex items-center gap-2">
                        <select
                            className="flex-1 p-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:border-blue-500 text-sm"
                            value={selectedViewType}
                            onChange={(e) => setSelectedViewType(e.target.value)}
                        >
                            <option value="">Select a view type...</option>
                            {Object.entries(viewComponents).map(([key]) => (
                                <option key={key} value={key}>
                                    {viewComponentNames[key] || key}
                                </option>
                            ))}
                        </select>
                        <button
                            onClick={handleCreateView}
                            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors text-sm"
                        >
                            Create
                        </button>
                        <button
                            onClick={() => {
                                setShowCreateView(false);
                                setSelectedViewType('');
                            }}
                            className="px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-md transition-colors text-sm"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}
            
            {/* Selected view content */}
            <div className="flex-1 overflow-auto">
                {selectedView || <div className="p-4 text-gray-400">Loading view...</div>}
            </div>
        </div>
    );
}