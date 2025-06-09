import React from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';

interface RecipeListProps {
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  parentEntityId: string;
}

interface RecipeChild {
  entity_id: string;
  data: {
    entityId: string; // Should be entity_id to match the parent prop
    name?: string;
  };
}

interface RecipeListItemRendererProps {
  child: RecipeChild;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
}

const RecipeListItemRenderer = React.memo(function RecipeListItemRenderer({ child, sendStrategyRequest, updateEntity }: RecipeListItemRendererProps) {
  const viewChildren = useRecoilValue(childrenByTypeSelector({ parentId: child.entity_id, type: EntityTypes.VIEW })) as any[];
  const listItemViewMeta = viewChildren.find((view) => view.data?.view_component_type === 'recipelistitem');
  const renderedListItemView = useRenderStoredView(listItemViewMeta?.entity_id, sendStrategyRequest, updateEntity);

  const handleDragStart = (event: React.DragEvent<HTMLDivElement>) => {
    console.log('RecipeListItemRenderer Drag Start, child.entity_id:', child.entity_id);
    if (!child.entity_id) {
      console.error("RecipeListItemRenderer: child.entity_id is undefined or null, aborting drag.");
      event.preventDefault();
      return;
    }
    event.dataTransfer.setData('text/plain', child.entity_id);
    event.dataTransfer.setData('application/json/recipe', JSON.stringify(child.data));
    event.dataTransfer.dropEffect = 'copy';

    // Create custom drag image
    const dragImage = document.createElement('div');
    dragImage.style.position = 'absolute';
    dragImage.style.top = '-1000px';
    dragImage.style.backgroundColor = 'rgba(55, 65, 81, 0.9)';
    dragImage.style.color = 'white';
    dragImage.style.padding = '4px 8px';
    dragImage.style.borderRadius = '4px';
    dragImage.style.fontSize = '0.875rem';
    dragImage.textContent = child.data?.name || 'Recipe';
    document.body.appendChild(dragImage);
    event.dataTransfer.setDragImage(dragImage, 0, 0);

    // Clean up the appended element
    setTimeout(() => {
      document.body.removeChild(dragImage);
    }, 0);
  };

  if (!renderedListItemView) {
    return <div>List Item View not available for {child.data.entityId}</div>;
  }

  const itemName = listItemViewMeta?.data?.name || 'Recipe Item';

  return (
    <div 
      className="truncate nodrag cursor-grab"
      title={itemName}
      draggable={true}
      onDragStart={handleDragStart}
    >
      {renderedListItemView}
    </div>
  );
});

export default function RecipeList({ sendStrategyRequest, updateEntity, parentEntityId }: RecipeListProps) {
  const recipeChildren = useRecoilValue(childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.RECIPE })) as any[];

  return (
    <div className="flex flex-col gap-2 nowheel h-full w-full p-2 overflow-y-auto">
      <div className="w-full">
        {recipeChildren.map((child) => (
          <RecipeListItemRenderer
            key={child.entity_id}
            child={child}
            sendStrategyRequest={sendStrategyRequest}
            updateEntity={updateEntity}
          />
        ))}
      </div>
    </div>
  );
}