import React from 'react';
import { useRecoilValue } from 'recoil';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView.js';
import { nodeSelectorFamily, childrenByTypeSelector } from '../../../../../../state/entitiesSelectors.js';

interface EntityRendererProps {
    entityData: any;
    sendStrategyRequest?: (request: any) => void;
    updateEntity?: (entityId: string, data: any) => void;
    showBorder?: boolean;
    className?: string;
}

interface SerializedEntityViewProps {
    viewData: any;
    parentData: any;
    sendStrategyRequest?: (request: any) => void;
    updateEntity?: (entityId: string, data: any) => void;
}

// New component to bridge to useRenderStoredView hook
interface RenderViewViaHookProps {
    viewEntityId: string;
    sendStrategyRequest?: (request: any) => void;
    updateEntity?: (entityId: string, data: any) => void;
}

const RenderViewViaHook: React.FC<RenderViewViaHookProps> = ({
    viewEntityId,
    sendStrategyRequest,
    updateEntity,
}) => {
    const renderedView = useRenderStoredView(viewEntityId, sendStrategyRequest, updateEntity);

    if (!renderedView) {
        return (
            <div className="text-yellow-400 text-sm p-2 bg-yellow-900/20 rounded w-64 h-64 flex items-center justify-center">
                <span className="text-center">View (ID: {viewEntityId}) could not be rendered.</span>
            </div>
        );
    }
    // Wrap the renderedView in a styled div for a square, icon-like preview
    return (
        <div className="w-64 h-64 overflow-hidden border border-gray-600 rounded-lg p-1 bg-gray-800/30 shadow-lg flex items-center justify-center">
            <div className="w-full h-full overflow-hidden relative">
                 {/* The actual rendered view, it will be clipped if larger than its container */}
                {renderedView}
            </div>
        </div>
    );
};

// Component to render entity information without a view
const DefaultEntityDisplay: React.FC<{ entity: any }> = ({ entity }) => {
    const displayName = entity.name || entity.entity_name || entity.entity_type || 'Unknown Entity';
    
    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-200">{displayName}</h3>
                <span className="text-xs text-gray-400 font-mono">{entity.entity_id}</span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div>
                    <span className="text-gray-400">Type:</span> 
                    <span className="ml-2 text-gray-200">{entity.entity_type || entity.entity_name}</span>
                </div>
                
                {entity.child_ids && entity.child_ids.length > 0 && (
                    <div>
                        <span className="text-gray-400">Children:</span> 
                        <span className="ml-2 text-gray-200">{entity.child_ids.length}</span>
                    </div>
                )}
                
                {entity.parent_ids && entity.parent_ids.length > 0 && (
                    <div>
                        <span className="text-gray-400">Parents:</span> 
                        <span className="ml-2 text-gray-200">{entity.parent_ids.length}</span>
                    </div>
                )}
            </div>

            {/* Show custom attributes */}
            {Object.entries(entity)
                .filter(([key, value]) => 
                    !['entity_id', 'entity_type', 'entity_name', 'child_ids', 'parent_ids', 
                      'strategy_requests', 'position', 'width', 'height', 'hidden', 
                      'class_path', 'meta_data'].includes(key) &&
                    value !== null && value !== undefined
                )
                .map(([key, value]) => (
                    <div key={key} className="text-sm border-t border-gray-700/50 pt-2">
                        <span className="text-gray-400 capitalize">{key.replace(/_/g, ' ')}:</span>
                        <div className="mt-1 ml-2 text-gray-200">
                            {typeof value === 'object' ? (
                                <pre className="text-xs bg-gray-800/50 p-2 rounded overflow-auto">
                                    {JSON.stringify(value, null, 2)}
                                </pre>
                            ) : (
                                <span>{String(value)}</span>
                            )}
                        </div>
                    </div>
                ))}
        </div>
    );
};

// New internal component that uses Recoil to find and render views
interface EntityDisplayAndRecoilViewsProps {
    entityId: string; // ID of the parent entity
    showBorder?: boolean;
    sendStrategyRequest?: (request: any) => void;
    updateEntity?: (entityId: string, data: any) => void;
}

const EntityDisplayAndRecoilViews: React.FC<EntityDisplayAndRecoilViewsProps> = ({
    entityId,
    showBorder,
    sendStrategyRequest,
    updateEntity,
}) => {
    // Assume selectors return an object with a 'data' property, or null/undefined if not found
    const entityState = useRecoilValue(nodeSelectorFamily(entityId)) as { data: any } | null;
    // Assume childrenByTypeSelector returns an array of such objects, or null/empty array
    const viewChildrenStates = useRecoilValue(childrenByTypeSelector({ parentId: entityId, type: 'view' })) as ({ data: any }[] | null);

    if (!entityState || !entityState.data) {
        return <div className="text-gray-500 text-sm p-4 text-center">Entity data not found for ID: {entityId}</div>;
    }
    
    const parentEntityData = entityState.data;
    const firstViewChild = viewChildrenStates && viewChildrenStates.length > 0 ? viewChildrenStates[0] : null;
    const entityName = parentEntityData.name || parentEntityData.entity_name || parentEntityData.entity_type || 'Unknown Entity';

    const containerClasses = showBorder 
        ? "border rounded-lg overflow-hidden"
        : "";

    const headerClasses = firstViewChild
        ? "bg-blue-800/20 border-blue-600/30 text-blue-300"
        : "bg-gray-700/30 border-gray-600/30 text-gray-300";
    
    return (
        <div
            className={`${containerClasses} ${firstViewChild ? 'border-blue-600/30 bg-blue-900/10' : 'border-gray-600/50 bg-gray-800/20'}`}
        >
            {showBorder && (
                <div className={`px-4 py-2 border-b ${headerClasses}`}>
                    <div className="flex items-center justify-between">
                        <h4 className="text-sm font-semibold">
                            {firstViewChild ? 'Entity with View' : 'Entity'}: {entityName}
                        </h4>
                        <span className="text-xs opacity-70">{parentEntityData.entity_id}</span>
                    </div>
                </div>
            )}
            <div className="p-4">
                {firstViewChild && firstViewChild.data?.entity_id ? (
                    <RenderViewViaHook
                        viewEntityId={firstViewChild.data.entity_id}
                        sendStrategyRequest={sendStrategyRequest}
                        updateEntity={updateEntity}
                    />
                ) : (
                    <DefaultEntityDisplay entity={parentEntityData} />
                )}
            </div>
        </div>
    );
};

// Main EntityRenderer component: Iterates entities from props and uses EntityDisplayAndRecoilViews for each
const EntityRenderer: React.FC<EntityRendererProps> = ({
    entityData, // This is the raw data, e.g., from chat "Entity Graph"
    sendStrategyRequest,
    updateEntity,
    showBorder = true,
    className = ""
}) => {
    if (!entityData) {
        return (
            <div className="text-gray-500 text-sm p-4 text-center">
                No entity data provided (EntityRenderer)
            </div>
        );
    }

    // Handle single entity or entity dictionary from the input prop
    // We are interested in the *IDs* of these entities primarily.
    const entitiesToProcess = entityData.entity_id 
        ? [entityData] // It's a single entity object
        : typeof entityData === 'object' && Object.keys(entityData).length > 0
            ? Object.values(entityData) // It's a dictionary of entities
            : []; // Fallback to empty array if format is unexpected

    if (entitiesToProcess.length === 0) {
         return (
            <div className="text-yellow-500 text-sm p-4 text-center">
                Could not parse entities from provided entityData (EntityRenderer).
            </div>
        );
    }
    
    return (
        <div className={`space-y-4 ${className}`}>
            {entitiesToProcess.map((entity: any, index: number) => {
                if (!entity || !entity.entity_id) {
                    // This check is important if entityData is a dictionary and some values are not valid entities
                    console.warn("EntityRenderer: Skipping an item without entity_id in entityData", entity);
                    return null; 
                }
                
                // Delegate rendering to the new component that uses Recoil for view discovery
                return (
                    <EntityDisplayAndRecoilViews
                        key={`recoil-view-${entity.entity_id}-${index}`}
                        entityId={entity.entity_id}
                        showBorder={showBorder}
                        sendStrategyRequest={sendStrategyRequest}
                        updateEntity={updateEntity}
                    />
                );
            }).filter(Boolean)}
        </div>
    );
};

export default EntityRenderer; 