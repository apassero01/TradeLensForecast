import React from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';

// Define types for props
interface AbstractChild {
  entityId: string;
  data: {
    entityId: string;
    // It's good practice to expect a name property, but handle its absence
    name?: string;
  };
  // Add other common properties of child if necessary
}

interface AbstractListItemRendererProps {
  child: AbstractChild;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  isExpanded: boolean;
  listItemViewName: string;
  expandedViewName?: string; // Optional: only needed if there's a distinct expanded view
  defaultItemName?: string; // Optional: fallback for item name
  createMissingViews?: boolean; // Whether to create missing views automatically
}

export default React.memo(function AbstractListItemRenderer({
  child,
  sendStrategyRequest,
  updateEntity,
  isExpanded,
  listItemViewName,
  expandedViewName,
  defaultItemName = 'Item' // Default fallback name
}: AbstractListItemRendererProps) {
  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: child.data.entityId, type: EntityTypes.VIEW })) as any[];

  // Find the specific views based on props
  const listItemViewMeta = viewChildren.find((view) => view.data?.view_component_type === listItemViewName);
  const expandedViewMeta = expandedViewName ? viewChildren.find((view) => view.data?.view_component_type === expandedViewName) : undefined;

  // Render the list item view always
  const renderedListItemView = useRenderStoredView(listItemViewMeta?.entity_id, sendStrategyRequest, updateEntity);
  // Render the expanded view only when needed and if defined
  const renderedExpandedView = useRenderStoredView(expandedViewMeta?.entity_id, sendStrategyRequest, updateEntity);

  if (!renderedListItemView) {
    return <div key={child.data.entityId}>{listItemViewName} view not available for {child.data.entityId}</div>;
  }

  // Attempt to get the name for the title attribute
  // Prioritize name from listItemViewMeta, then child.data, then defaultItemName
  const itemName = listItemViewMeta?.data?.name || child.data?.name || defaultItemName;

  const shouldShowExpandedView = isExpanded && expandedViewName && expandedViewMeta;

  return (
    <div key={child.data.entityId} className="mb-1 flex flex-col">
      {/* List Item View - Wrapped for truncation and title */}
      <div className="truncate" title={itemName}>
        {renderedListItemView}
      </div>

      {/* Expanded View (conditionally shown) */}
      {shouldShowExpandedView && renderedExpandedView && (
        <div className="mt-1 p-1 overflow-y-auto max-h-[500px]"> {/* Consider making max-h configurable */}
          {renderedExpandedView}
        </div>
      )}
      {/* Error message if expanded view is expected but fails to render */}
      {shouldShowExpandedView && !renderedExpandedView && (
        <div className="mt-1 p-1 border border-red-500 overflow-y-auto max-h-[500px]">
          {expandedViewName} component exists but could not be rendered.
        </div>
      )}
    </div>
  );
}); 