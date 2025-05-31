import React from 'react';
import { IoFastFood, IoClose, IoEyeOutline } from 'react-icons/io5';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface RecipeListItemProps {
  data?: RecipeListItemData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string; 
  parentEntityId: string;
}

interface RecipeListItemData {
  name: string;
}

export default function RecipeListItem({ data, sendStrategyRequest, parentEntityId, viewEntityId }: RecipeListItemProps) {
  const handleDelete = () => {
    sendStrategyRequest(StrategyRequests.removeChild(viewEntityId, viewEntityId));
  };

  return (
    <div className="flex items-center justify-between p-4 border-b border-gray-700 hover:bg-gray-600 cursor-pointer w-full h-full bg-gray-700 space-x-3">
      <div className="flex items-center space-x-3 min-w-0">
        {/* @ts-ignore */}
        <IoFastFood className="text-gray-300" size={24} />
        <h3 className="text-xl font-semibold text-gray-100 overflow-hidden text-ellipsis whitespace-nowrap">
          {data?.name || 'Untitled Recipe'}
        </h3>
      </div>
      <div className="flex items-center space-x-1"> {/* Wrapper for buttons */}
        <button
          onClick={() => sendStrategyRequest(StrategyRequests.hideEntity(parentEntityId, false))}
          className="text-gray-400 hover:text-blue-500 p-1"
          aria-label="Show recipe"
        >
          {/* @ts-ignore */}
          <IoEyeOutline size={20} />
        </button>
      </div>
    </div>
  );
}