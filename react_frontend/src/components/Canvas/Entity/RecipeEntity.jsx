import React from 'react';
import EntityNodeBase from './EntityNodeBase'; // Adjust the import path as necessary
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../state/entitiesSelectors'; // Adjust the import path as necessary
import { EntityTypes } from './EntityEnum'; // Adjust the import path as necessary
import useRenderStoredView from '../../../hooks/useRenderStoredView'; // Adjust the import path as necessary

/**
 * React component for rendering a Recipe entity node.
 * It finds a child View entity designated for displaying instructions
 * and renders it within the node.
 */
function RecipeEntity({ data, updateEntity, sendStrategyRequest }) {

  // Select child View entities linked to this RecipeEntity
  const viewEntities = useRecoilValue(childrenByTypeSelector({ parentId: data.entityId, type: EntityTypes.VIEW }));

  // Find the specific View entity intended for displaying instructions.
  // We'll assume it has a specific 'view_component_type' like 'instructions_display'.
  // This might need adjustment based on your actual view configuration.
  const instructionsViewEntity = viewEntities.find(view =>
    view.data?.view_component_type === "recipeinstructions"
  );

  const instructionsViewEntityIdToUse = instructionsViewEntity?.entity_id;

  // Use the custom hook to render the content of the found View entity
  const instructionsView = useRenderStoredView(instructionsViewEntityIdToUse, sendStrategyRequest, updateEntity);

  // Define the content rendering logic for the node body
  const renderContent = () => (
    <div className="flex flex-col h-full w-full">
      {/* Label for the instructions view section */}
      <h4 className="text-xs text-gray-500 mt-2 mb-1 px-2 flex-shrink-0">Instructions View:</h4>
      {/* Container for the rendered view */}
      <div className="nodrag border border-gray-300 rounded p-1 m-1 bg-gray-50 flex-grow min-h-0 overflow-auto">
        {/* Render the instructions view if found, otherwise show a placeholder */}
        {instructionsView ? instructionsView : <p className="text-xs text-gray-400 italic">No instructions view available.</p>}
      </div>
    </div>
  );

  // Render the main node structure using EntityNodeBase
  return (
    <EntityNodeBase
      data={data} // Pass entity data to the base node
      updateEntity={updateEntity} // Pass update function
      
    >
      {/* Pass a function that returns the result of renderContent */}
      {(baseProps) => renderContent()}
    </EntityNodeBase>
  );
}

// Memoize the component for performance optimization
export default React.memo(RecipeEntity);
