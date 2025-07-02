import React, { useState, useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily, childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import { 
  IoFolder, 
  IoFolderOpen, 
  IoDocumentText, 
  IoChevronForward, 
  IoChevronDown,
  IoAdd,
  IoTrash,
  IoPencil,
  IoEllipsisVertical
} from 'react-icons/io5';

interface FileTreeProps {
  data?: FileTreeData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
  handleSelect?: (entityId: string) => void;
}

interface FileTreeData {
  name?: string;
  is_folder?: boolean;
}

interface TreeNodeProps {
  entityId: string;
  level: number;
  onSelect: (entityId: string) => void;
  selectedId: string | null;
  expandedNodes: Set<string>;
  toggleExpanded: (entityId: string) => void;
  sendStrategyRequest: (strategyRequest: any) => void;
  parentEntityId: string;
  currentDraggedId: string | null;
  setCurrentDraggedId: (id: string | null) => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({ 
  entityId, 
  level, 
  onSelect, 
  selectedId,
  expandedNodes,
  toggleExpanded,
  sendStrategyRequest,
  parentEntityId,
  currentDraggedId,
  setCurrentDraggedId
}) => {
  const entity = useRecoilValue(nodeSelectorFamily(entityId)) as any;
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [showRenameInput, setShowRenameInput] = useState(false);
  const [newName, setNewName] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  // Get document children if this is a folder
  const documentChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: entityId, type: EntityTypes.DOCUMENT })
  ) as any[];

  if (!entity || !entity.data) return null;

  const { name, docName, document_type, file_type, path } = entity.data;
  const displayName = docName || name || 'Unnamed';
  const isFolder = document_type === 'directory';
  const isExpanded = expandedNodes.has(entityId);
  const isSelected = selectedId === entityId;

  const handleDragStart = (e: React.DragEvent) => {
    console.log('Drag started for:', entityId, displayName);
    setCurrentDraggedId(entityId);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', entityId);
    e.dataTransfer.setData('sourceParentId', parentEntityId); // Store source parent
    
    // Add visual feedback
    (e.currentTarget as HTMLElement).style.opacity = '0.5';
  };

  const handleDragEnd = (e: React.DragEvent) => {
    console.log('Drag ended at:', e.clientX, e.clientY);
    
    // Reset visual feedback
    (e.currentTarget as HTMLElement).style.opacity = '1';
    
    // Check if dropped outside of file tree
    const dropTarget = document.elementFromPoint(e.clientX, e.clientY);
    const fileTreeContainer = dropTarget?.closest('.file-tree-root');
    
    console.log('Drop target:', dropTarget);
    console.log('Inside file tree?', !!fileTreeContainer);
    
    if (!fileTreeContainer) {
      // Dropped outside - unhide and position the entity
      const canvas = dropTarget?.closest('[data-canvas="true"]') || document.querySelector('.relative.w-full.h-full');
      console.log('Canvas found:', !!canvas);
      
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        sendStrategyRequest(StrategyRequests.builder()
          .withStrategyName('SetAttributesStrategy')
          .withTargetEntity(entityId)
          .withParams({
            attribute_map: { 
              hidden: false,
              position: { x: Math.round(x), y: Math.round(y) }
            }
          })
          .build());
        
        console.log('Entity dropped on canvas:', entityId, 'at', x, y);
      }
    }
    
    setCurrentDraggedId(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (isFolder && currentDraggedId && currentDraggedId !== entityId) {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(true);
    }
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    
    if (isFolder && currentDraggedId && currentDraggedId !== entityId) {
      // Move the dragged entity to this folder
      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('ChangeParentStrategy')
        .withTargetEntity(currentDraggedId)
        .withParams({
          parent_entity_id: entityId
        })
        .build());
      
      console.log('Entity moved:', currentDraggedId, 'to folder:', entityId);
    }
  };

  const handleClick = () => {
    if (isFolder) {
      toggleExpanded(entityId);
    } else {
      onSelect(entityId);
    }
  };

  const handleCreateFile = (isNewFolder: boolean) => {
    const fileName = window.prompt(`Enter ${isNewFolder ? 'folder' : 'file'} name:`);
    if (!fileName) return;

    const fileType = isNewFolder ? null : window.prompt('Enter file type (e.g., python, markdown, javascript):') || 'text';

    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('CreateEntityStrategy')
      .withTargetEntity(entityId)
      .withParams({
        entity_class: 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity',
        initial_attributes: {
          name: fileName,
          document_type: isNewFolder ? 'directory' : 'file',
          file_type: isNewFolder ? null : fileType,
          text: isNewFolder ? null : ''
        }
      })
      .build());

    setShowContextMenu(false);
  };

  const handleRename = () => {
    if (newName && newName !== name) {
      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(entityId)
        .withParams({
          attribute_map: { name: newName }
        })
        .build());
    }
    setShowRenameInput(false);
    setNewName('');
  };

  const handleUpdateDocName = () => {
    const newDocName = window.prompt('Enter document display name:', docName || '');
    if (newDocName !== null) {
      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(entityId)
        .withParams({
          attribute_map: { docName: newDocName }
        })
        .build());
    }
  };

  const handleDelete = () => {
    if (window.confirm(`Are you sure you want to delete "${displayName}"?`)) {
      // Clear selection if this document is selected
      const selectionAttribute = `ide_selected_by_${parentEntityId}`;
      if (entity.data?.[selectionAttribute]) {
        sendStrategyRequest(StrategyRequests.builder()
          .withStrategyName('SetAttributesStrategy')
          .withTargetEntity(entityId)
          .withParams({
            attribute_map: { [selectionAttribute]: null }
          })
          .build());
      }

      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('RemoveEntityStrategy')
        .withTargetEntity(entityId)
        .build());
    }
    setShowContextMenu(false);
  };

  const getFileIcon = () => {
    if (isFolder) {
      return isExpanded ? (
        /* @ts-ignore */
        <IoFolderOpen className="text-blue-400" />
      ) : (
        /* @ts-ignore */
        <IoFolder className="text-blue-400" />
      );
    }
    
    // Different colors based on file type or file extension from name
    const fileExtension = name?.split('.').pop()?.toLowerCase();
    const iconColors: { [key: string]: string } = {
      python: 'text-yellow-400',
      py: 'text-yellow-400',
      javascript: 'text-yellow-300',
      js: 'text-yellow-300',
      typescript: 'text-blue-300',
      ts: 'text-blue-300',
      tsx: 'text-blue-300',
      markdown: 'text-gray-300',
      md: 'text-gray-300',
      json: 'text-orange-300',
      html: 'text-orange-400',
      css: 'text-blue-400',
    };
    
    const color = iconColors[file_type || ''] || iconColors[fileExtension || ''] || 'text-gray-400';
    return (
      /* @ts-ignore */
      <IoDocumentText className={color} />
    );
  };

  return (
    <div className="select-none">
      <div 
        className={`flex items-center px-2 py-1 hover:bg-gray-700 cursor-pointer group relative ${
          isSelected ? 'bg-gray-700' : ''
        } ${isDragOver ? 'bg-blue-900/30 border border-blue-500' : ''}`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={handleClick}
        onContextMenu={(e) => {
          e.preventDefault();
          setShowContextMenu(true);
        }}
        draggable={true}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Expand/Collapse Icon - Always show for folders, even if empty */}
        {isFolder ? (
          <span className="mr-1 text-gray-500">
            {isExpanded ? (
              /* @ts-ignore */
              <IoChevronDown size={12} />
            ) : (
              /* @ts-ignore */
              <IoChevronForward size={12} />
            )}
          </span>
        ) : <span className="mr-1 w-3" />}

        {/* File/Folder Icon */}
        <span className="mr-2 cursor-move">{getFileIcon()}</span>

        {/* Name */}
        {showRenameInput ? (
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onBlur={handleRename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleRename();
              if (e.key === 'Escape') {
                setShowRenameInput(false);
                setNewName('');
              }
            }}
            className="flex-1 px-1 bg-gray-800 border border-blue-500 rounded text-sm"
            autoFocus
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span className="flex-1 text-sm truncate">
            {displayName}
            {docName && <span className="text-gray-500 ml-1 text-xs">({name})</span>}
            {/* Show child count for debugging */}
            {isFolder && process.env.NODE_ENV === 'development' && (
              <span className="text-gray-600 ml-2 text-xs">({documentChildren.length})</span>
            )}
          </span>
        )}

        {/* Context Menu Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowContextMenu(!showContextMenu);
          }}
          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-600 rounded"
        >
          {/* @ts-ignore */}
          <IoEllipsisVertical size={14} />
        </button>

        {/* Context Menu */}
        {showContextMenu && (
          <div
            className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-600 rounded shadow-lg py-1 z-10 min-w-[160px]"
            onMouseLeave={() => setShowContextMenu(false)}
          >
            {(isDirectory || hasChildren) && (
              <>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCreateFile(false);
                  }}
                  className="w-full px-3 py-1 text-left hover:bg-gray-700 text-sm flex items-center gap-2"
                >
                  {/* @ts-ignore */}
                  <IoAdd size={14} /> New File
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCreateFile(true);
                  }}
                  className="w-full px-3 py-1 text-left hover:bg-gray-700 text-sm flex items-center gap-2"
                >
                  {/* @ts-ignore */}
                  <IoAdd size={14} /> New Folder
                </button>
                <div className="border-t border-gray-700 my-1" />
              </>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setNewName(name || '');
                setShowRenameInput(true);
                setShowContextMenu(false);
              }}
              className="w-full px-3 py-1 text-left hover:bg-gray-700 text-sm flex items-center gap-2"
            >
              {/* @ts-ignore */}
              <IoPencil size={14} /> Rename
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleUpdateDocName();
                setShowContextMenu(false);
              }}
              className="w-full px-3 py-1 text-left hover:bg-gray-700 text-sm flex items-center gap-2"
            >
              {/* @ts-ignore */}
              <IoPencil size={14} /> Set Display Name
            </button>
            <div className="border-t border-gray-700 my-1" />
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete();
              }}
              className="w-full px-3 py-1 text-left hover:bg-gray-700 text-sm text-red-400 flex items-center gap-2"
            >
              {/* @ts-ignore */}
              <IoTrash size={14} /> Delete
            </button>
          </div>
        )}
      </div>

      {/* Render children if expanded - show empty message if no children */}
      {isFolder && isExpanded && (
        <div>
          {documentChildren.length > 0 ? (
            documentChildren.map((child) => (
              <TreeNode
                key={child.entity_id}
                entityId={child.entity_id}
                level={level + 1}
                onSelect={onSelect}
                selectedId={selectedId}
                expandedNodes={expandedNodes}
                toggleExpanded={toggleExpanded}
                sendStrategyRequest={sendStrategyRequest}
                parentEntityId={parentEntityId}
                currentDraggedId={currentDraggedId}
                setCurrentDraggedId={setCurrentDraggedId}
              />
            ))
          ) : (
            <div 
              className="text-gray-500 text-xs italic" 
              style={{ paddingLeft: `${(level + 1) * 16 + 8}px` }}
            >
              Empty folder
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default function FileTree({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
  handleSelect,
}: FileTreeProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set([parentEntityId]));
  const [currentDraggedId, setCurrentDraggedId] = useState<string | null>(null);

  // Get all document children of the parent entity
  const documentChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.DOCUMENT })
  ) as any[];

  const toggleExpanded = (entityId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(entityId)) {
      newExpanded.delete(entityId);
    } else {
      newExpanded.add(entityId);
    }
    setExpandedNodes(newExpanded);
  };

  const handleSelectDocument = (entityId: string) => {
    if (handleSelect) {
      handleSelect(entityId);
      setSelectedId(entityId);
    } else {
      handleSelectInternal(entityId);
    }
  };
  
  const handleSelectInternal = (entityId: string) => {

    // Set new selection
    setSelectedId(entityId);
    
    // Mark the document as selected by this IDE instance
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(entityId)
      .withParams({
        attribute_map: { [`ide_selected_by_${parentEntityId}`]: true }
      })
      .build());
    
    console.log('Selected document:', entityId);
  };

  // Handle drops on the FileTree container (for cross-FileTree transfers)
  const handleFileTreeDragOver = (e: React.DragEvent) => {
    // During dragOver, we can't access the data, so just allow all drops
    // We'll filter in the drop handler
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleFileTreeDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const draggedEntityId = e.dataTransfer.getData('text/plain');
    const sourceParentId = e.dataTransfer.getData('sourceParentId');
    
    // Handle drop from different FileTree
    if (draggedEntityId && sourceParentId && sourceParentId !== parentEntityId) {
      console.log('Cross-FileTree transfer:', draggedEntityId, 'from', sourceParentId, 'to', parentEntityId);
      
      // Move the entity to this FileTree's parent
      sendStrategyRequest(StrategyRequests.builder()
        .withStrategyName('ChangeParentStrategy')
        .withTargetEntity(draggedEntityId)
        .withParams({
          parent_entity_id: parentEntityId
        })
        .build());
    }
  };

  if (!data) {
    return (
      <div className="p-4 text-gray-500">
        Loading file tree...
      </div>
    );
  }

  return (
    <div 
      className="nodrag py-2 overflow-y-auto nowheel max-h-full file-tree-root"
      data-parent-entity-id={parentEntityId}
      onDragOver={handleFileTreeDragOver}
      onDrop={handleFileTreeDrop}
    >
      {documentChildren.length === 0 ? (
        <div className="p-4 text-center text-gray-500 text-sm">
          <p>No files or folders yet</p>
          <button
            onClick={() => {
              const fileName = window.prompt('Enter file name:');
              if (!fileName) return;
              
              const isFolder = window.confirm('Create as folder?');
              const fileType = isFolder ? null : window.prompt('Enter file type:') || 'text';

              sendStrategyRequest(StrategyRequests.builder()
                .withStrategyName('CreateEntityStrategy')
                .withTargetEntity(parentEntityId)
                .withParams({
                  entity_class: 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity',
                  initial_attributes: {
                    name: fileName,
                    document_type: isFolder ? 'directory' : 'file',
                    file_type: isFolder ? null : fileType,
                    text: isFolder ? null : ''
                  }
                })
                .build());
            }}
            className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs flex items-center gap-1 mx-auto"
          >
            {/* @ts-ignore */}
            <IoAdd /> Create First File
          </button>
        </div>
      ) : (
        documentChildren.map((child) => (
          <TreeNode
            key={child.entity_id}
            entityId={child.entity_id}
            level={0}
            onSelect={handleSelectDocument}
            selectedId={selectedId}
            expandedNodes={expandedNodes}
            toggleExpanded={toggleExpanded}
            sendStrategyRequest={sendStrategyRequest}
            parentEntityId={parentEntityId}
            currentDraggedId={currentDraggedId}
            setCurrentDraggedId={setCurrentDraggedId}
          />
        ))
      )}
    </div>
  );
} 