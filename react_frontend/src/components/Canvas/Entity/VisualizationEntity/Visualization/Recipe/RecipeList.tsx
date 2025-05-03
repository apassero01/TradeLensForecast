import React, {useEffect, useRef} from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import RecipeListItemRenderer from './RecipeListItemRenderer';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface RecipeListProps {
  data?: RecipeListData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
}

interface RecipeListData {
  entity_id: string;
}

export default function RecipeList({ data, sendStrategyRequest, updateEntity }: RecipeListProps) {
  const parent_id = data?.entity_id ?? null;
  const recipeChildren = useRecoilValue(childrenByTypeSelector({ parentId: parent_id, type: EntityTypes.RECIPE })) as any[];

  const processedChildrenRef = useRef({});
  useEffect(() => {
    if (recipeChildren?.length > 0) {
      const updatesToMake = [];
      recipeChildren.forEach((child) => {
        if (!processedChildrenRef.current[child.entity_id]) {
          updatesToMake.push(child);
          processedChildrenRef.current[child.entity_id] = true;
        }
      });
      if (updatesToMake.length > 0 && updateEntity) {
        updatesToMake.forEach((child) => {
          sendStrategyRequest(StrategyRequests.hideEntity(child.entity_id, true));
          updateEntity(child.entity_id, {
            hidden: true,
          });
        });
      }
    }
  }, [recipeChildren, updateEntity]);

  if (!data) {
    return <div>Loading or error...</div>;
  }

  return (
    <div className="flex flex-col gap-2 nowheel h-full">
      <div className="overflow-y-auto h-full">
        {recipeChildren.map((child) => (
          <RecipeListItemRenderer
            key={child.entityId}
            child={child}
            sendStrategyRequest={sendStrategyRequest}
            updateEntity={updateEntity}
          />
        ))}
      </div>
    </div>
  );
}