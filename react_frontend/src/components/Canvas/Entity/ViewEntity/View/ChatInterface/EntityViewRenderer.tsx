import React from 'react';
import useEntityView from '../../../../../../hooks/useEntityView';

interface EntityViewRendererProps {
    entityId: string;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
}

export default function EntityViewRenderer({ 
    entityId, 
    sendStrategyRequest, 
    updateEntity 
}: EntityViewRendererProps) {
    // Use the hook to get the view for this entity
    const entityView = useEntityView(
        entityId, 
        sendStrategyRequest, 
        updateEntity, 
        {}, 
        undefined // Let it find any view type
    );

    if (!entityView) {
        // Fallback to a simple display
        return (
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Entity ID:</div>
                <div className="text-sm font-mono text-gray-300">{entityId}</div>
            </div>
        );
    }

    return <>{entityView}</>;
}