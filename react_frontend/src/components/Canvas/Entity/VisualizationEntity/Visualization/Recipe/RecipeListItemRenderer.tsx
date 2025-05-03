import React, { useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';

// Define types for props - adjust based on actual entity structure if needed
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
}

export default function RecipeListItemRenderer({ child, sendStrategyRequest, updateEntity }: RecipeListItemRendererProps) {
  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: child.data.entityId, type: EntityTypes.VIEW })) as any[];

  // Find the specific views
  const listItemViewMeta = viewChildren.find((view) => view.data?.view_component_type === 'recipelistitem');
  const instructionViewMeta = viewChildren.find((view) => view.data?.view_component_type === 'recipeinstructions');

  // State to track if instructions should be shown
  const [showInstructions, setShowInstructions] = useState<boolean>(false);

  // Render the list item view always
  const renderedListItemView = useRenderStoredView(listItemViewMeta?.entity_id, sendStrategyRequest, updateEntity);
  // Render the instruction view only when needed
  const renderedInstructionView = useRenderStoredView(instructionViewMeta?.entity_id, sendStrategyRequest, updateEntity);

  const handleItemClick = () => {
    // Toggle the instruction view visibility
    // Only toggle if an instruction view actually exists
    if (instructionViewMeta?.entity_id) {
      setShowInstructions(!showInstructions);
    }
  };

  if (!renderedListItemView) {
    // Handle case where the list item view is not found or not rendered
    return <div key={child.data.entityId}>List Item View not available for {child.data.entityId}</div>;
  }

  return (
    // Main container for the list item and potentially instructions
    // Use child.data.entityId as the key, assuming it's the recipe's unique ID
    <div key={child.data.entityId} className="mb-1">
      {/* List Item View (always shown) */}
      <div onClick={handleItemClick} className="cursor-pointer">
        {renderedListItemView}
      </div>

      {/* Instruction View (conditionally shown) */}
      {showInstructions && renderedInstructionView && (
        <div className="mt-1 p-1 max-h-[500px] overflow-y-auto">
          {renderedInstructionView}
        </div>
      )}
       {showInstructions && !renderedInstructionView && (
        <div className="mt-1 p-1 max-h-[300px] overflow-y-auto border border-red-500">
          Instruction View component exists but could not be rendered.
        </div>
      )}
    </div>
  );
} 