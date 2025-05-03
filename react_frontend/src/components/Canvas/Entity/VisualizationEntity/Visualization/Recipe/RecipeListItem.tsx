import React from 'react';
import { IoFastFood } from 'react-icons/io5';

interface RecipeListItemProps {
  data?: RecipeListItemData;
}

interface RecipeListItemData {
  name: string;
}

export default function RecipeListItem({ data }: RecipeListItemProps) {
  return (
    <div className="flex items-center p-4 border-b border-gray-700 hover:bg-gray-600 cursor-pointer w-full h-full bg-gray-700 space-x-3">
      {/* @ts-ignore */}
      <IoFastFood className="text-gray-300" size={24} />
      <h3 className="text-xl font-semibold text-gray-100">{data?.name || 'Untitled Recipe'}</h3>
    </div>
  );
}