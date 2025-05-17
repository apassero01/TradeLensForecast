import React from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';

// Define types for props
interface RecipeChild {
  entityId: string;
  data: {
    entityId: string;
  };
  // Add other properties of child if necessary
}

interface RecipeListItemRendererProps {
  child: RecipeChild;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  isExpanded: boolean; // Changed from viewMode
}

export default React.memo(function RecipeListItemRenderer({ child, sendStrategyRequest, updateEntity, isExpanded }: RecipeListItemRendererProps) {
  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: child.data.entityId, type: EntityTypes.VIEW })) as any[];

  // Find the specific views
  const listItemViewMeta = viewChildren.find((view) => view.data?.view_component_type === 'recipelistitem');
  const instructionViewMeta = viewChildren.find((view) => view.data?.view_component_type === 'recipeinstructions');

  // Render the list item view always
  const renderedListItemView = useRenderStoredView(listItemViewMeta?.entity_id, sendStrategyRequest, updateEntity);
  // Render the instruction view only when needed (based on isExpanded prop)
  const renderedInstructionView = useRenderStoredView(instructionViewMeta?.entity_id, sendStrategyRequest, updateEntity);

  if (!renderedListItemView) {
    // Handle case where the list item view is not found or not rendered
    return <div key={child.data.entityId}>List Item View not available for {child.data.entityId}</div>;
  }

  // Attempt to get the name for the title attribute
  const itemName = listItemViewMeta?.data?.name || 'Recipe Item'; // Fallback text

  // Determine if instructions should be shown based on isExpanded prop
  const shouldShowInstructions = isExpanded; // New logic

  return (
    // Removed grid-specific styling, using mb-1 for spacing
    <div key={child.data.entityId} className="mb-1 flex flex-col">
      {/* List Item View - Wrapped for truncation and title */}
      <div className="truncate" title={itemName}>
        {renderedListItemView}
      </div>

      {/* Instruction View (conditionally shown based on isExpanded) */}
      {shouldShowInstructions && renderedInstructionView && (
        // Added max-h-[500px]
        <div className="mt-1 p-1 overflow-y-auto max-h-[500px]">
          {renderedInstructionView}
        </div>
      )}
      {/* Error message if instruction view exists but fails to render */}
       {shouldShowInstructions && instructionViewMeta && !renderedInstructionView && (
         // Also add max-h to the error view for consistency?
        <div className="mt-1 p-1 border border-red-500 overflow-y-auto max-h-[500px]">
          Instruction View component exists but could not be rendered.
        </div>
      )}
    </div>
  );
}); 