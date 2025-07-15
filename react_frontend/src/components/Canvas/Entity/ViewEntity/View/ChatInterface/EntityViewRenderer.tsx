import React, { useState } from 'react';
import useEntityView from '../../../../../../hooks/useEntityView';

interface EntityViewRendererProps {
    entityId: string;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    isIcon?: boolean;
    onIconClick?: () => void;
}

export default function EntityViewRenderer({ 
    entityId, 
    sendStrategyRequest, 
    updateEntity,
    isIcon = false,
    onIconClick
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
        const fallbackContent = (
            <div className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                <div className="text-xs text-gray-400 mb-1">Entity ID:</div>
                <div className="text-sm font-mono text-gray-300">{entityId}</div>
            </div>
        );

        if (isIcon) {
            return (
                <div 
                    className="w-12 h-12 overflow-hidden cursor-pointer hover:scale-110 transition-transform"
                    onClick={onIconClick}
                    title={`Entity: ${entityId}`}
                >
                    <div className="scale-[0.2] origin-top-left w-[240px] h-[240px]">
                        {fallbackContent}
                    </div>
                </div>
            );
        }

        return fallbackContent;
    }

    if (isIcon) {
        return (
            <div 
                className="w-12 h-12 overflow-hidden cursor-pointer hover:scale-110 transition-transform border border-gray-600 rounded-lg"
                onClick={onIconClick}
                title={`Click to view entity: ${entityId}`}
            >
                <div className="scale-[0.2] origin-top-left w-[240px] h-[240px]">
                    {entityView}
                </div>
            </div>
        );
    }

    return <>{entityView}</>;
}