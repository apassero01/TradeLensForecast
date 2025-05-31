import React from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';

interface MealPlannerDashboardProps {
  data?: MealPlannerDashboardData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface MealPlannerDashboardData {
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

export default function MealPlannerDashboard({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: MealPlannerDashboardProps) {
  // Get all view children to find existing views
  const viewChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.VIEW })
  ) as any[];

  // Get all recipe children for count
  const recipeChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.RECIPE })
  ) as any[];

  // Find existing views (these should be created by the backend)
  const mealPlanView = viewChildren.find(
    (view) => view.data?.view_component_type === 'mealplan'
  );

  const recipeListView = viewChildren.find(
    (view) => view.data?.view_component_type === 'recipelist'
  );

  // Render the views using the existing hook
  const renderedMealPlanView = useRenderStoredView(
    mealPlanView?.entity_id,
    sendStrategyRequest,
    updateEntity
  );

  const renderedRecipeListView = useRenderStoredView(
    recipeListView?.entity_id,
    sendStrategyRequest,
    updateEntity
  );

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Loading meal planner...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-gray-800 text-white overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-600">
        <h1 className="text-2xl font-bold text-white mb-2">
          {data.name || 'Meal Planner'}
        </h1>
        {data.week_start_date && (
          <p className="text-gray-400 text-sm">
            Week starting: {new Date(data.week_start_date).toLocaleDateString()}
          </p>
        )}
      </div>

      {/* Content - Recipes above Meal Plan */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Recipes Section Wrapper */}
        {/* This section's height is determined by the RecipeList component itself. */}
        {/* RecipeList requests its own height via a strategy. */}
        <div className="flex-shrink-0 border-b border-gray-600">
          <div className="p-2"> {/* Reduced padding for section header */}
            <h2 className="text-lg font-semibold text-white mb-1"> {/* Reduced margin-bottom */}
              Available Recipes ({recipeChildren.length})
            </h2>
          </div>
          {/* Container for RecipeList. RecipeList component has h-full and manages its own content. */}
          {/* No max-h here, allowing RecipeList to expand to its requested height. */}
          <div> 
            {renderedRecipeListView ? (
              renderedRecipeListView
            ) : (
              <div className="flex items-center justify-center h-32 text-gray-500 p-2">
                <div className="text-center">
                  <div className="text-2xl mb-2">📋</div>
                  <p>Recipe list loading...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Meal Plan Section - Takes remaining space */}
        {/* Removed the redundant "Weekly Meal Plan" header */}
        <div className="flex-1 overflow-auto p-2"> 
          {renderedMealPlanView ? (
            renderedMealPlanView
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <div className="text-4xl mb-4">📅</div>
                <p>Meal plan view loading...</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 