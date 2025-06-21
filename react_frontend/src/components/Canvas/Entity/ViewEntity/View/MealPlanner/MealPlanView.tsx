import React, { useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';
import { IoClose, IoCheckbox, IoSquareOutline } from 'react-icons/io5';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface MealPlanViewProps {
  data?: MealPlanViewData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface MealPlanViewData {
  name: string;
  week_start_date: string;
  monday: string[];
  tuesday: string[];
  wednesday: string[];
  thursday: string[];
  friday: string[];
  saturday: string[];
  sunday: string[];
}

const DAYS_OF_WEEK = [
  { key: 'monday', label: 'Monday' },
  { key: 'tuesday', label: 'Tuesday' },
  { key: 'wednesday', label: 'Wednesday' },
  { key: 'thursday', label: 'Thursday' },
  { key: 'friday', label: 'Friday' },
  { key: 'saturday', label: 'Saturday' },
  { key: 'sunday', label: 'Sunday' },
];

export default function MealPlanView({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: MealPlanViewProps) {
  const [showRecipeModal, setShowRecipeModal] = useState(false);
  const [selectedDay, setSelectedDay] = useState<string>('');
  const [selectedRecipes, setSelectedRecipes] = useState<Set<string>>(new Set());
  const [draggedOverDay, setDraggedOverDay] = useState<string | null>(null); // For drag-over visual feedback

  // Get all recipe children of the meal plan
  const recipeChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.RECIPE })
  ) as any[];

  // Create a map of recipe ID to recipe entity for quick lookup
  const recipeMap = React.useMemo(() => {
    const map: { [key: string]: any } = {};
    recipeChildren.forEach((recipe) => {
      map[recipe.entity_id] = recipe;
    });
    return map;
  }, [recipeChildren]);

  // Function to open the recipe selection modal
  const openRecipeModal = (day: string) => {
    setSelectedDay(day);
    setSelectedRecipes(new Set());
    setShowRecipeModal(true);
  };

  // Function to toggle recipe selection
  const toggleRecipeSelection = (recipeId: string) => {
    const newSelected = new Set(selectedRecipes);
    if (newSelected.has(recipeId)) {
      newSelected.delete(recipeId);
    } else {
      newSelected.add(recipeId);
    }
    setSelectedRecipes(newSelected);
  };

  // Function to add selected recipes to the day
  const addSelectedRecipes = () => {
    selectedRecipes.forEach((recipeId) => {
      // TODO: Call AddRecipeToDayStrategy for each selected recipe
      console.log(`Adding recipe ${recipeId} to ${selectedDay}`);
      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('AddRecipeToDayStrategy')
        .withTargetEntity(parentEntityId)
        .withParams({ day: selectedDay, recipe_id: recipeId })
        .build());
    });
    setShowRecipeModal(false);
    setSelectedRecipes(new Set());
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>, dayKey: string) => {
    event.preventDefault();
    // Dynamically set dropEffect based on the drag source type
    if (Array.from(event.dataTransfer.types).includes('application/json/mealplanitem')) {
      event.dataTransfer.dropEffect = 'move';
    } else {
      event.dataTransfer.dropEffect = 'copy';
    }
    setDraggedOverDay(dayKey);
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>, dayKey: string) => {
    event.preventDefault(); // Necessary for some browsers
    setDraggedOverDay(dayKey);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    setDraggedOverDay(null);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>, dayKey: string) => {
    event.preventDefault();
    setDraggedOverDay(null);

    // Try to get data for an internal move first
    const mealPlanItemDataString = event.dataTransfer.getData('application/json/mealplanitem');

    if (mealPlanItemDataString) {
      try {
        const { recipeId: RId, sourceDayKey } = JSON.parse(mealPlanItemDataString);
        console.log(`MealPlanView Internal Drop: Parsed data - targetDayKey='${dayKey}', recipeId='${RId}', sourceDayKey='${sourceDayKey}'`);

        if (RId && sourceDayKey && sourceDayKey !== dayKey) {
          console.log(`Attempting to move recipe ${RId} from ${sourceDayKey} to ${dayKey}`);
          // 1. Remove from the source day
          sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('RemoveRecipeFromDayStrategy')
            .withTargetEntity(parentEntityId)
            .withParams({ day: sourceDayKey, recipe_id: RId })
            .withAddToHistory(false) // Consistent with your previous change
            .build());

          // 2. Add to the target day
          sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('AddRecipeToDayStrategy')
            .withTargetEntity(parentEntityId)
            .withParams({ day: dayKey, recipe_id: RId })
            .build());
        } else if (sourceDayKey === dayKey) {
          console.log("Recipe dropped onto the same day. No action taken.");
        } else {
          console.warn("MealPlanView Internal Drop: Invalid RId or sourceDayKey, or sourceDayKey is the same as target. No move action taken.", { RId, sourceDayKey, targetDayKey: dayKey });
        }
      } catch (error) {
        console.error("Error parsing meal plan item data or processing internal drop:", error);
        // Fallback to text/plain if JSON parsing fails or data is malformed
        const recipeIdFromText = event.dataTransfer.getData('text/plain');
        if (recipeIdFromText) {
          console.log(`MealPlanView Drop (fallback from error): dayKey='${dayKey}', recipeId='${recipeIdFromText}'`);
          sendStrategyRequest(StrategyRequests.builder()
            .withStrategyName('AddRecipeToDayStrategy')
            .withTargetEntity(parentEntityId)
            .withParams({ day: dayKey, recipe_id: recipeIdFromText })
            .build());
        }
      }
      
    } else {
      // Fallback: Drag from external source (e.g., RecipeList)
      const recipeId = event.dataTransfer.getData('text/plain');
      console.log(`MealPlanView External Drop: dayKey='${dayKey}', recipeId='${recipeId}'`);

      if (recipeId) {
        sendStrategyRequest(StrategyRequests.builder()
          .withStrategyName('AddRecipeToDayStrategy')
          .withTargetEntity(parentEntityId)
          .withParams({ day: dayKey, recipe_id: recipeId })
          .build());
      }
    }
  };

  // Function to render recipe list items for a specific day
  const renderRecipesForDay = (recipeIds: string[], dayKey: string) => {
    if (!recipeIds || recipeIds.length === 0) {
      return (
        <div className="text-center text-gray-500 py-4 text-sm">
          No recipes planned
        </div>
      );
    }

    return recipeIds.map((recipeId) => {
      const recipe = recipeMap[recipeId];

      // Defensive check for falsy recipeId
      if (!recipeId) {
        console.warn('MealPlanView renderRecipesForDay: Encountered a falsy recipeId. Skipping render for this item.');
        return null; // Or some placeholder if you prefer
      }

      if (!recipe) {
        return (
          <div key={recipeId} className="text-red-400 text-sm p-2 bg-gray-700 rounded-md relative border border-red-500">
            <span>Recipe not found: {recipeId.substring(0, 8)}...</span>
            <button
              onClick={() => {
                sendStrategyRequest(StrategyRequests.builder()
                  .withStrategyName('RemoveRecipeFromDayStrategy')
                  .withTargetEntity(parentEntityId) // This is the MealPlan entity ID
                  .withParams({ day: dayKey, recipe_id: recipeId })
                  .withAddToHistory(false)
                  .build());
              }}
              className="absolute top-1 right-1 text-gray-400 hover:text-red-400 p-1 rounded-full transition-colors nodrag"
              aria-label="Remove missing recipe"
            >
              {/* @ts-ignore */}
              <IoClose size={16} />
            </button>
          </div>
        );
      }

      return <RecipeItemRenderer key={recipeId} recipe={recipe} sendStrategyRequest={sendStrategyRequest} updateEntity={updateEntity} parentEntityId={parentEntityId} dayKey={dayKey} />;
    });
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Loading meal plan...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-gray-800 text-white p-4 overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">
          {data.name || 'Meal Plan'}
        </h1>
        {data.week_start_date && (
          <p className="text-gray-400 text-sm">
            Week starting: {new Date(data.week_start_date).toLocaleDateString()}
          </p>
        )}
      </div>

      {/* Calendar Grid */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-7 gap-4 overflow-y-auto">
        {DAYS_OF_WEEK.map((day) => {
          const dayRecipes = data[day.key as keyof MealPlanViewData] as string[];
          const isDraggedOver = draggedOverDay === day.key;
          
          return (
            <div
              key={day.key}
              onDragOver={(e) => handleDragOver(e, day.key)}
              onDragEnter={(e) => handleDragEnter(e, day.key)}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, day.key)}
              className={`bg-gray-700 rounded-lg border border-gray-600 flex flex-col min-h-[300px] transition-all duration-150 ${
                isDraggedOver ? 'ring-2 ring-blue-500 border-blue-500 bg-gray-600' : ''
              }`}
            >
              {/* Day Header */}
              <div className="bg-gray-600 rounded-t-lg p-3 border-b border-gray-500">
                <h3 className="font-semibold text-lg text-center text-white">
                  {day.label}
                </h3>
              </div>

              {/* Recipes Container */}
              <div className="flex-1 p-2 overflow-y-auto">
                <div className="space-y-2">
                  {renderRecipesForDay(dayRecipes, day.key)}
                </div>
              </div>

              {/* Add Recipe Button */}
              <div className="p-2 border-t border-gray-600">
                <button
                  onClick={() => openRecipeModal(day.key)}
                  className="w-full py-2 px-3 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <span className="text-lg">+</span>
                  Add Recipe
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recipe Selection Modal */}
      {showRecipeModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg border border-gray-600 w-full max-w-2xl max-h-[80vh] flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-600">
              <h2 className="text-xl font-bold text-white">
                Add Recipes to {DAYS_OF_WEEK.find(d => d.key === selectedDay)?.label}
              </h2>
              <button
                onClick={() => setShowRecipeModal(false)}
                className="text-gray-400 hover:text-white p-1"
              >
                {/* @ts-ignore */}
                <IoClose size={24} />
              </button>
            </div>

            {/* Recipe List */}
            <div className="flex-1 overflow-y-auto p-4">
              {recipeChildren.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  No recipes available
                </div>
              ) : (
                <div className="space-y-2">
                  {recipeChildren.map((recipe) => (
                    <div
                      key={recipe.entity_id}
                      onClick={() => toggleRecipeSelection(recipe.entity_id)}
                      className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg border border-gray-600 hover:bg-gray-600 cursor-pointer transition-colors"
                    >
                      {/* Checkbox */}
                      <div className="text-blue-400">
                        {selectedRecipes.has(recipe.entity_id) ? (
                          /* @ts-ignore */
                          <IoCheckbox size={20} />
                        ) : (
                          /* @ts-ignore */
                          <IoSquareOutline size={20} />
                        )}
                      </div>
                      
                      {/* Recipe Info */}
                      <div className="flex-1">
                        <h3 className="text-white font-medium">
                          {recipe.data?.name || 'Unnamed Recipe'}
                        </h3>

                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="flex items-center justify-between p-4 border-t border-gray-600">
              <div className="text-gray-400 text-sm">
                {selectedRecipes.size} recipe{selectedRecipes.size !== 1 ? 's' : ''} selected
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowRecipeModal(false)}
                  className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={addSelectedRecipes}
                  disabled={selectedRecipes.size === 0}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-md transition-colors"
                >
                  Add {selectedRecipes.size > 0 ? `${selectedRecipes.size} ` : ''}Recipe{selectedRecipes.size !== 1 ? 's' : ''}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Component to render individual recipe items
function RecipeItemRenderer({
  recipe,
  sendStrategyRequest,
  updateEntity,
  parentEntityId,
  dayKey,
}: {
  recipe: any;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  parentEntityId: string;
  dayKey: string;
}) {
  // Get the recipe's view children to find the list item view
  const viewChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: recipe.entity_id, type: EntityTypes.VIEW })
  ) as any[];

  // Find the recipe list item view
  const recipeListItemView = viewChildren.find(
    (view) => view.data?.view_component_type === 'recipelistitem'
  );

  // Render the stored view
  const renderedView = useRenderStoredView(
    recipeListItemView?.entity_id,
    sendStrategyRequest,
    updateEntity
  );

  const handleDragStart = (event: React.DragEvent<HTMLDivElement>) => {
    // Prevent drag if the target is an element with the 'nodrag' class
    // const targetElement = event.target as HTMLElement;
    // if (targetElement.closest('.nodrag')) { // Check for .nodrag class
    //   event.preventDefault();
    //   console.log("Drag prevented on .nodrag element");
    //   return;
    // }

    const dragData = JSON.stringify({ recipeId: recipe.entity_id, sourceDayKey: dayKey });
    event.dataTransfer.setData('application/json/mealplanitem', dragData);
    event.dataTransfer.effectAllowed = 'move';
    console.log(`RecipeItemRenderer DragStart: recipeId='${recipe.entity_id}', sourceDayKey='${dayKey}'`);

    // Create custom drag image (copied from RecipeListItemRenderer.tsx)
    const dragImage = document.createElement('div');
    dragImage.style.position = 'absolute';
    dragImage.style.top = '-1000px'; // Position off-screen
    dragImage.style.backgroundColor = 'rgba(55, 65, 81, 0.9)'; // bg-gray-700 with opacity
    dragImage.style.color = 'white';
    dragImage.style.padding = '4px 8px'; // p-1 px-2
    dragImage.style.borderRadius = '4px'; // rounded
    dragImage.style.fontSize = '0.875rem'; // text-sm
    dragImage.textContent = recipe.data?.name || 'Recipe Item'; // Use recipe.data.name or fallback
    document.body.appendChild(dragImage);
    event.dataTransfer.setDragImage(dragImage, 10, 10); // Small offset from cursor

    // Clean up the appended element
    setTimeout(() => {
      if (document.body.contains(dragImage)) {
        document.body.removeChild(dragImage);
      }
    }, 0);
  };

  const handleRemoveRecipe = () => {
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('RemoveRecipeFromDayStrategy')
      .withTargetEntity(parentEntityId)
      .withParams({ day: dayKey, recipe_id: recipe.entity_id })
      .build());
  };

  if (!renderedView) {
    return (
      <div
        draggable={true}
        onDragStart={handleDragStart}
        className="bg-gray-600 rounded p-2 text-sm text-gray-300 relative cursor-grab nodrag"
      >
        <div className="font-medium">{recipe.data?.name || 'Unnamed Recipe'}</div>
        <div className="text-xs text-gray-400 mt-1">View not available</div>
        <button
          onClick={handleRemoveRecipe}
          className="absolute top-1 right-1 text-gray-400 hover:text-red-500 p-1 rounded-full transition-colors"
          aria-label="Remove recipe"
        >
          {/* @ts-ignore */}
          <IoClose size={18} />
        </button>
      </div>
    );
  }

  return (
    <div
      draggable={true}
      onDragStart={handleDragStart}
      className="transform scale-95 origin-top relative group cursor-grab nodrag"
    >
      {renderedView}
      <button
        onClick={handleRemoveRecipe}
        className="absolute top-1 right-1 text-gray-400 hover:text-red-500 bg-gray-700 bg-opacity-50 hover:bg-opacity-75 p-1 rounded-full transition-all opacity-0 group-hover:opacity-100"
        aria-label="Remove recipe"
      >
        {/* @ts-ignore */}
        <IoClose size={18} />
      </button>
    </div>
  );
} 