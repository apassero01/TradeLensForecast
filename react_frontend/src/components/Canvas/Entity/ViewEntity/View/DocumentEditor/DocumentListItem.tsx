import React from 'react';
import { FiFile, FiFileText, FiFolder, FiEyeOff } from 'react-icons/fi';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface DocumentListItemProps {
  data?: {
    name?: string;
    path?: string;
    type?: string;
    content?: string;
    fileType?: string;
    text?: string;
    docName?: string;
    is_folder?: boolean;
  };
  sendStrategyRequest: (strategyRequest: any) => void;
  parentEntityId: string;
}

/**
 * DocumentListItem component displays a compact view of a document for use in lists
 * It shows a document icon, name, and potentially a preview of content
 */
export default function DocumentListItem({ data, sendStrategyRequest, parentEntityId }: DocumentListItemProps) {
  // Use docName if available, otherwise fall back to name or path
  const displayName = data?.docName || data?.name || (data?.path ? data?.path.split('/').pop() : null) || 'Untitled';
  
  // Show actual name in parentheses if docName is being used
  const showActualName = data?.docName && data?.name && data.docName !== data.name;

  // Get file extension from name if available (for icon selection)
  const fileName = data?.name || displayName;
  const fileExt = !data?.is_folder && fileName.includes('.') ? fileName.split('.').pop()?.toLowerCase() : '';

  // Get a very brief preview of the content if available
  const contentText = data?.text || data?.content;
  const contentPreview = contentText
    ? contentText.substring(0, 50) + (contentText.length > 50 ? '...' : '')
    : 'No content';

  // Helper to select icon based on file type or extension
  const getIcon = () => {
    if (data?.is_folder) {
      // @ts-ignore
      return <FiFolder className="mr-2 text-blue-500" />;
    }
    
    switch (data?.fileType || fileExt) {
      case 'md':
      case 'markdown':
        // @ts-ignore
        return <FiFileText className="mr-2 text-blue-600" />;
      case 'txt':
      case 'text':
      case 'doc':
      case 'docx':
        // @ts-ignore
        return <FiFileText className="mr-2 text-gray-600" />;
      case 'py':
      case 'python':
        // @ts-ignore
        return <FiFile className="mr-2 text-yellow-600" />;
      case 'js':
      case 'javascript':
      case 'ts':
      case 'typescript':
        // @ts-ignore
        return <FiFile className="mr-2 text-yellow-500" />;
      default:
        // @ts-ignore
        return <FiFile className="mr-2 text-gray-600" />;
    }
  };

  return (
    <div className="flex flex-col p-2 bg-white border rounded-md shadow-sm hover:shadow transition-shadow">
      <div className="flex items-center mb-1">
        {getIcon()}
        <div className="flex-1 truncate">
          <span className="font-medium text-gray-800" title={displayName}>
            {displayName}
          </span>
          {showActualName && (
            <span className="text-xs text-gray-500 ml-1">({data.name})</span>
          )}
        </div>
        <button
          type="button"
          onClick={() => {
            sendStrategyRequest(StrategyRequests.hideEntity(parentEntityId, false));
          }}
          className="ml-2 p-1 hover:bg-gray-200 rounded"
        >
          {/* @ts-ignore */}
          <FiEyeOff size={16} className="text-gray-500" />
        </button>
      </div>

      {!data?.is_folder && contentText && (
        <div className="text-xs text-gray-500 line-clamp-2 mt-1">
          {contentPreview}
        </div>
      )}
    </div>
  );
} 