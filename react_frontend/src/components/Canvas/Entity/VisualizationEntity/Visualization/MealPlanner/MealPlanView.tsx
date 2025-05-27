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

  // Function to render recipe list items for a specific day
  const renderRecipesForDay = (recipeIds: string[]) => {
    if (!recipeIds || recipeIds.length === 0) {
      return (
        <div className="text-center text-gray-500 py-4 text-sm">
          No recipes planned
        </div>
      );
    }

    return recipeIds.map((recipeId) => {
      const recipe = recipeMap[recipeId];
      if (!recipe) {
        return (
          <div key={recipeId} className="text-red-400 text-sm p-2">
            Recipe not found: {recipeId}
          </div>
        );
      }

      return <RecipeItemRenderer key={recipeId} recipe={recipe} sendStrategyRequest={sendStrategyRequest} updateEntity={updateEntity} />;
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
          
          return (
            <div
              key={day.key}
              className="bg-gray-700 rounded-lg border border-gray-600 flex flex-col min-h-[300px]"
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
                  {renderRecipesForDay(dayRecipes)}
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
}: {
  recipe: any;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
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

  if (!renderedView) {
    return (
      <div className="bg-gray-600 rounded p-2 text-sm text-gray-300">
        <div className="font-medium">{recipe.data?.name || 'Unnamed Recipe'}</div>
        <div className="text-xs text-gray-400 mt-1">View not available</div>
      </div>
    );
  }

  return (
    <div className="transform scale-95 origin-top">
      {renderedView}
    </div>
  );
} 