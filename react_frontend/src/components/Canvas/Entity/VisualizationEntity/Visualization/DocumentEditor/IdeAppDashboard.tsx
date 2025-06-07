import React, { useEffect, useMemo, useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector, nodeSelectorFamily, allEntitiesSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';
import { IoFolder, IoDocumentText, IoSearch, IoClose } from 'react-icons/io5';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';
import AdvancedDocumentEditor from './AdvancedDocumentEditor';

interface IdeAppDashboardProps {
  data?: IdeAppDashboardData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface IdeAppDashboardData {
  name?: string;
}

interface DocumentTab {
  id: string;
  name: string;
  documentData: any;
  entityData: any;
}

export default function IdeAppDashboard({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: IdeAppDashboardProps) {
  // State for managing tabs
  const [openTabs, setOpenTabs] = useState<DocumentTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);

  // Get all view children to find existing views
  const viewChildren = useRecoilValue(
    childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.VIEW })
  ) as any[];

  // Get all entities to work with
  const allEntities = useRecoilValue(allEntitiesSelector) as any[];
  
  // Create a lookup map for quick entity access
  const entityMap = useMemo(() => {
    const map = new Map();
    if (Array.isArray(allEntities)) {
      allEntities.forEach(entity => {
        if (entity?.data?.entity_id) {
          map.set(entity.data.entity_id, entity.data);
        }
      });
    }
    return map;
  }, [allEntities]);

  // Function to recursively collect all documents from an entity tree
  const collectAllDocuments = useMemo(() => {
    const collect = (entityId: string, visited = new Set<string>()): any[] => {
      if (visited.has(entityId)) {
        return [];
      }
      
      visited.add(entityId);
      const entityData = entityMap.get(entityId);
      
      if (!entityData) {
        return [];
      }
      
      const documents: any[] = [];
      
      // If this entity is a document, include it
      if (entityData.entity_type === EntityTypes.DOCUMENT) {
        documents.push(entityData);
      }
      
      // Recursively process all children
      if (entityData.child_ids && Array.isArray(entityData.child_ids)) {
        entityData.child_ids.forEach((childId: string) => {
          const childDocuments = collect(childId, new Set(visited));
          documents.push(...childDocuments);
        });
      }
      
      return documents;
    };
    
    return collect;
  }, [entityMap, allEntities]);

  // Get all documents recursively from the parent entity
  const allDocuments = useMemo(() => {
    try {
      return collectAllDocuments(parentEntityId);
    } catch (error) {
      console.warn('Error collecting documents recursively:', error);
      return [];
    }
  }, [parentEntityId, collectAllDocuments]);

  // Find all selected documents - look for ones with ide_selected_by_<parentEntityId> attribute
  const selectedDocuments = useMemo(() => {
    const selectionAttribute = `ide_selected_by_${parentEntityId}`;
    return allDocuments.filter(doc => doc[selectionAttribute] === true);
  }, [allDocuments, parentEntityId]);

  // Sync open tabs with selected documents
  useEffect(() => {
    const newTabs: DocumentTab[] = [];
    
    selectedDocuments.forEach(doc => {
      const existingTab = openTabs.find(tab => tab.id === doc.entity_id);
      if (existingTab) {
        newTabs.push({
          ...existingTab,
          documentData: doc,
          name: doc.docName || doc.name || 'Untitled'
        });
      } else {
        newTabs.push({
          id: doc.entity_id,
          name: doc.docName || doc.name || 'Untitled',
          documentData: doc,
          entityData: null
        });
      }
    });

    const tabsChanged = 
      newTabs.length !== openTabs.length ||
      newTabs.some(tab => !openTabs.find(existing => existing.id === tab.id));

    if (tabsChanged) {
      setOpenTabs(newTabs);
      
      // Determine which tabs were newly added in this cycle
      // These are tabs in `newTabs` whose IDs were not in `openTabs` (from the closure, before this update)
      const newlyAddedTabs = newTabs.filter(nt => !openTabs.find(ot => ot.id === nt.id));

      if (newlyAddedTabs.length > 0) {
        // If new tabs were added, make the last one added the active tab
        setActiveTabId(newlyAddedTabs[newlyAddedTabs.length - 1].id);
      } else if (!activeTabId || !newTabs.find(tab => tab.id === activeTabId)) {
        // If no new tabs were added, but the current activeTabId is invalid 
        // (e.g., its tab was closed or was never set), set to the first available tab or null.
        setActiveTabId(newTabs.length > 0 ? newTabs[0].id : null);
      }
      // If no new tabs were added and activeTabId is still valid, it remains the active tab.
    }
  }, [selectedDocuments, parentEntityId, openTabs, activeTabId]);

  // Get entity data for the active tab
  const activeTab = openTabs.find(tab => tab.id === activeTabId);
  const activeDocumentEntity = useRecoilValue(
    nodeSelectorFamily(activeTab?.id || null)
  ) as any;

  // Tab management functions
  const closeTab = (tabId: string, event?: React.MouseEvent) => {
    if (event) {
      event.stopPropagation();
    }
    
    const newTabs = openTabs.filter(tab => tab.id !== tabId);
    setOpenTabs(newTabs);
    
    // If we closed the active tab, switch to another tab
    // send strategy request to set the attribute to false
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(tabId)
      .withParams({
        attribute_map: { [`ide_selected_by_${parentEntityId}`]: false }
      })
      .build());
    
    if (activeTabId === tabId) {
      if (newTabs.length > 0) {
        // Find the tab that was next to the closed one, or the first one
        const closedTabIndex = openTabs.findIndex(tab => tab.id === tabId);
        const newActiveIndex = Math.min(closedTabIndex, newTabs.length - 1);
        setActiveTabId(newTabs[newActiveIndex].id);
      } else {
        setActiveTabId(null);
      }
    }
    
    // Note: We don't update the backend selection state here
    // This allows the tab to potentially reopen if the document remains selected in the backend
  };

  const switchToTab = (tabId: string) => {
    setActiveTabId(tabId);
  };

  // Find existing views (these should be created by the backend)
  const fileTreeView = viewChildren.find(
    (view) => view.data?.view_component_type === 'file_tree'
  );

  const searchView = viewChildren.find(
    (view) => view.data?.view_component_type === 'document_search'
  );

  // Create missing views on mount
  useEffect(() => {
    const createMissingViews = () => {
      // Create file tree view if it doesn't exist
      if (!fileTreeView) {
        const fileTreeRequest = StrategyRequests.builder()
          .withStrategyName('CreateEntityStrategy')
          .withTargetEntity(parentEntityId)
          .withParams({
            entity_class: 'shared_utils.entities.view_entity.ViewEntity.ViewEntity',
            initial_attributes: {
              parent_attributes: {
                "name": "name",
                "is_folder": "is_folder"
              },
              view_component_type: 'file_tree',
              hidden: true
            }
          })
          .build();
        
        sendStrategyRequest(fileTreeRequest);
      }

      // Create search view if it doesn't exist
      if (!searchView) {
        const searchViewRequest = StrategyRequests.builder()
          .withStrategyName('CreateEntityStrategy')
          .withTargetEntity(parentEntityId)
          .withParams({
            entity_class: 'shared_utils.entities.view_entity.ViewEntity.ViewEntity',
            initial_attributes: {
              parent_attributes: {},
              view_component_type: 'document_search',
                hidden: false
            }
          })
          .build();
        
        sendStrategyRequest(searchViewRequest);
      }
    };

    // Small delay to avoid creating views multiple times during initial render
    const timer = setTimeout(createMissingViews, 100);
    return () => clearTimeout(timer);
  }, [parentEntityId, fileTreeView, searchView, sendStrategyRequest]);

  // Render the views using the existing hook
  const renderedFileTreeView = useRenderStoredView(
    fileTreeView?.entity_id,
    sendStrategyRequest,
    updateEntity
  );

  const renderedSearchView = useRenderStoredView(
    searchView?.entity_id,
    sendStrategyRequest,
    updateEntity
  );

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        Loading IDE...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full bg-gray-900 text-white overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-700">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          {IoFolder({ className: "text-blue-400" }) as React.JSX.Element}
          {data.name || 'Document IDE'}
        </h1>
        <p className="text-gray-400 text-sm mt-1">
          {allDocuments.length} documents
          {openTabs.length > 0 && (
            <span className="ml-2">
              • {openTabs.length} open tab{openTabs.length !== 1 ? 's' : ''}
              {activeTab && (
                <span className="ml-2">
                  • Editing: {activeTab.name}
                </span>
              )}
            </span>
          )}
        </p>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - File Tree */}
        <div className="w-80 border-r border-gray-700 flex flex-col">
          {/* Search Section */}
          <div className="flex-shrink-0 border-b border-gray-700">
            <div className="p-3">
              <h2 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2">
                {IoSearch({}) as React.JSX.Element}
                Search Documents
              </h2>
            </div>
            <div className="px-3 pb-3">
              {renderedSearchView ? (
                renderedSearchView
              ) : (
                <div className="flex items-center justify-center h-20 text-gray-500">
                  <p className="text-sm">
                    {searchView ? 'Search loading...' : 'Creating search view...'}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* File Tree Section */}
          <div className="flex-1 overflow-hidden flex flex-col">
            <div className="p-3 border-b border-gray-700">
              <h2 className="text-sm font-semibold text-gray-400 flex items-center gap-2">
                {IoFolder({}) as React.JSX.Element}
                File Explorer
              </h2>
            </div>
            <div className="flex-1 overflow-auto nowheel pb-4">
              {renderedFileTreeView ? (
                renderedFileTreeView
              ) : (
                <div className="flex items-center justify-center h-32 text-gray-500 p-4">
                  <div className="text-center">
                    <div className="text-3xl mb-2">📁</div>
                    <p>{fileTreeView ? 'File tree loading...' : 'Creating file tree...'}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Editor Area */}
        <div className="flex-1 bg-gray-800 flex flex-col">
          {/* Tab Bar */}
          {openTabs.length > 0 && (
            <div className="flex-shrink-0 bg-gray-800 border-b border-gray-700">
              <div className="flex overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
                {openTabs.map((tab) => (
                  <div
                    key={tab.id}
                    className={`
                      flex items-center px-4 py-2 border-r border-gray-700 cursor-pointer
                      min-w-0 max-w-48 group transition-colors duration-150
                      ${activeTabId === tab.id 
                        ? 'bg-gray-700 text-white border-b-2 border-blue-400' 
                        : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
                      }
                    `}
                    onClick={() => switchToTab(tab.id)}
                    title={tab.name}
                  >
                    {IoDocumentText({ className: "flex-shrink-0 mr-2 text-sm" }) as React.JSX.Element}
                    <span className="truncate text-sm font-medium">
                      {tab.name}
                    </span>
                    <button
                      className={`
                        flex-shrink-0 ml-2 p-1 rounded hover:bg-gray-600 
                        opacity-0 group-hover:opacity-100 transition-opacity duration-150
                        ${activeTabId === tab.id ? 'opacity-60 hover:opacity-100' : ''}
                      `}
                      onClick={(e) => closeTab(tab.id, e)}
                      title="Close tab"
                    >
                      {IoClose({ className: "text-xs" }) as React.JSX.Element}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Editor Content */}
          <div className="flex-1">
            {activeTab && activeDocumentEntity ? (
              <AdvancedDocumentEditor
                data={activeDocumentEntity.data}
                sendStrategyRequest={sendStrategyRequest}
                updateEntity={updateEntity}
                viewEntityId={viewEntityId}
                parentEntityId={activeTab.id}
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500">
                  <div className="text-6xl mb-4">
                    {IoDocumentText({}) as React.JSX.Element}
                  </div>
                  <p className="text-lg mb-2">No document selected</p>
                  <p className="text-sm">Select a file from the explorer to start editing</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 