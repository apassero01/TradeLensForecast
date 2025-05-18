import React from 'react';
import { FiFile, FiFileText, FiFolder } from 'react-icons/fi';

interface EditorListItemProps {
  data?: {
    name?: string;
    path?: string;
    type?: string;
    content?: string;
    fileType?: string;
  };
}

/**
 * EditorListItem component displays a compact view of a document for use in lists
 * It shows a document icon, name, and potentially a preview of content
 */
export default function EditorListItem({ data }: EditorListItemProps) {
  // Extract filename from path or use provided name, or fallback to "Untitled"
  const displayName = data?.name || (data?.path ? data?.path.split('/').pop() : null) || 'Untitled';
  
  // Get file extension if available (for icon selection)
  const fileExt = displayName.includes('.') ? displayName.split('.').pop()?.toLowerCase() : '';
  
  // Get a very brief preview of the content if available
  const contentPreview = data?.content 
    ? data.content.substring(0, 50) + (data.content.length > 50 ? '...' : '') 
    : 'No content';

  // Choose appropriate icon based on type or extension
  const getIcon = () => {
    // @ts-ignore
    if (data?.type === 'folder') return <FiFolder className="mr-2 text-blue-500" />;
    
    // File type icons - could be expanded with more specific icons for different file types
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
      </div>
      
      {data?.content && (
        <div className="text-xs text-gray-500 line-clamp-2 mt-1">
          {contentPreview}
        </div>
      )}
    </div>
  );
} 