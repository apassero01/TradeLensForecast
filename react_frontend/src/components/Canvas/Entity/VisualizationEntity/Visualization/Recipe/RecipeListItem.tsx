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

export default function RecipeListItem({ data, sendStrategyRequest, parentEntityId }: RecipeListItemProps) {
  const handleDelete = () => {
    // Assuming StrategyRequests.removeEntity is the correct method
    // and it takes the entityId of the item to be removed.
    // In this context, we want to remove the recipe item itself,
    // so we should use an ID associated with *this* item, not its parent.
    // However, the user specifically asked for parentEntityId to be used.
    // This might be an error in the request, as typically one would remove the item itself.
    // For now, I will follow the user's request and use parentEntityId.
    // If this recipe item has its own entity_id, that would typically be used here.
    // Let's assume 'data.entity_id' or a similar prop would hold the actual ID of this item.
    // For now, sticking to the user's direct request:
    sendStrategyRequest(StrategyRequests.removeEntity(parentEntityId));
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
        <button
          onClick={handleDelete}
          className="text-gray-400 hover:text-red-500 p-1"
          aria-label="Remove recipe"
        >
          {/* @ts-ignore */}
          <IoClose size={20} />
        </button>
      </div>
    </div>
  );
}