import React from 'react';
import { EntityTypes } from '../../../../Entity/EntityEnum'; // Make sure this path is correct
import AbstractList from '../Abstract/AbstractList';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

// Define props for EditorList, similar to RecipeList but can be simpler if many defaults from AbstractList are used
interface EditorListProps {
  data?: { // Data from the parent visualization entity (e.g., width, height)
    entity_id: string;
    width?: number;
    height?: number;
  };
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string; // The entity ID of this EditorList view itself
  parentEntityId: string; // The entity ID of the parent whose documents are to be listed
}

// Define some default layout properties for the document grid
const DOCUMENT_ITEM_WIDTH_EXPANDED = 350;  // Default width for a document item in the grid
const DOCUMENT_ITEM_HEIGHT_EXPANDED = 450; // Default height for a document item in the grid
const DOCUMENT_GRID_COLS_EXPANDED = 3;     // Default number of columns in the grid
const DOCUMENT_GAP_EXPANDED = 16;          // Default gap between items in the grid (in pixels)
const DOCUMENT_HEADER_FOOTER_HEIGHT = 60;  // Estimated height for list controls (button, padding)

/**
 * EditorList displays a grid of documents with:
 * - For the list item view: document_summary_view (our custom EditorListItem component)
 * - For the expanded view: editor (the standard editor component)
 */
export default function EditorList({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
}: EditorListProps) {
  return (
    <AbstractList
      data={data}
      sendStrategyRequest={sendStrategyRequest}
      updateEntity={updateEntity}
      viewEntityId={viewEntityId}
      parentEntityId={parentEntityId}
      entityType={EntityTypes.DOCUMENT} // Specify that we are listing DOCUMENT entities
      listItemViewName="document_summary_view" // View type for collapsed state - our custom view
      expandedViewName="editor" // View type for expanded items - the standard editor component
      defaultItemName="Document" // Fallback name for items
      initialExpandAllState={true} // Start in "expanded" (grid) mode
      // Configure grid appearance
      itemWidthExpanded={DOCUMENT_ITEM_WIDTH_EXPANDED}
      itemHeightExpanded={DOCUMENT_ITEM_HEIGHT_EXPANDED}
      gridColsExpanded={DOCUMENT_GRID_COLS_EXPANDED}
      gapExpanded={DOCUMENT_GAP_EXPANDED}
      headerFooterHeight={DOCUMENT_HEADER_FOOTER_HEIGHT}
      // Always create missing views when needed to ensure proper display
      createMissingViews={true}
    />
  );
} 