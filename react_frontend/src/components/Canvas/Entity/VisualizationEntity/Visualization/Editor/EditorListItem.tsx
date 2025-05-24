import React from 'react';
import { FiFile, FiFileText, FiFolder, FiEyeOff } from 'react-icons/fi';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
interface EditorListItemProps {
  data?: {
    name?: string;
    path?: string;
    type?: string;
    content?: string;
    fileType?: string;
  };
    sendStrategyRequest: (strategyRequest: any) => void;
    parentEntityId: string;
}

/**
 * EditorListItem component displays a compact view of a document for use in lists
 * It shows a document icon, name, and potentially a preview of content
 */
export default function EditorListItem({ data, sendStrategyRequest, parentEntityId }: EditorListItemProps) {
  // Extract filename from path or use provided name, or fallback to "Untitled"
  const displayName = data?.name || (data?.path ? data?.path.split('/').pop() : null) || 'Untitled';

  // Get file extension if available (for icon selection)
  const fileExt = displayName.includes('.') ? displayName.split('.').pop()?.toLowerCase() : '';

  // Get a very brief preview of the content if available
  const contentPreview = data?.content
    ? data.content.substring(0, 50) + (data.content.length > 50 ? '...' : '')
    : 'No content';

  // Helper to select icon based on file extension
  const getIcon = () => {
    switch (fileExt) {
      case 'md':
      case 'txt':
      case 'doc':
      case 'docx':
        // @ts-ignore
        return <FiFileText className="mr-2 text-gray-600" />;
      default:
        // @ts-ignore
        return <FiFile className="mr-2 text-gray-600" />;
    }
  };

  return (
    <div className="flex flex-col p-2 bg-white border rounded-md shadow-sm hover:shadow transition-shadow">
      <div className="flex items-center mb-1">
        {getIcon()}
        <span className="font-medium text-gray-800 truncate" title={displayName}>
          {displayName}
        </span>
        <button
          type="button"
          onClick={() => {
            sendStrategyRequest(StrategyRequests.hideEntity(parentEntityId, false));
          }}
          className="ml-auto p-1 hover:bg-gray-200 rounded"
        >
          {/* @ts-ignore */}
          <FiEyeOff size={16} className="text-gray-500" />
        </button>
      </div>

      {data?.content && (
        <div className="text-xs text-gray-500 line-clamp-2 mt-1">
          {contentPreview}
        </div>
      )}
    </div>
  );
}
