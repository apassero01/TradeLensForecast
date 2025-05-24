import React, { useEffect, useRef, useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import AbstractListItemRenderer from './AbstractListItemRenderer';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface AbstractListProps {
  data?: AbstractListData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
  entityType: EntityTypes; // To specify what type of children to list
  listItemViewName: string; // To specify the view for list items
  expandedViewName?: string; // Optional: view for expanded items
  defaultItemName?: string; // Optional: fallback item name for renderer
  initialExpandAllState?: boolean; // Optional: initial state for expandAll
  // Optional: Configuration for resizing behavior
  itemHeightCollapsed?: number;
  itemGapCollapsed?: number;
  itemWidthCollapsed?: number;
  maxListItemsBeforeScroll?: number; // How many items to show before estimating scroll, or for fallback
  itemHeightExpanded?: number;
  itemWidthExpanded?: number;
  gridColsExpanded?: number;
  gapExpanded?: number;
  headerFooterHeight?: number; // Combined height for headers, footers, buttons etc.
  enableAutoResize?: boolean; // Control auto-resize feature
  hideChildrenInitially?: boolean; // Control if children are hidden initially
  createMissingViews?: boolean; // Whether to automatically create missing views
}

interface AbstractListData {
  entity_id: string;
  width?: number;
  height?: number;
}

// Default dimensions and behavior constants (can be overridden by props)
const DEFAULT_ITEM_HEIGHT_COLLAPSED = 48;
const DEFAULT_ITEM_GAP_COLLAPSED = 4;
const DEFAULT_ITEM_WIDTH_COLLAPSED = 400;
const DEFAULT_MAX_LIST_ITEMS_SCROLL = 5; // Show 5 items + buffer, then rely on scrollHeight or fallback
const DEFAULT_FALLBACK_MAX_ITEMS = 40; // Max items for fallback height if scrollHeight not available

const DEFAULT_ITEM_HEIGHT_EXPANDED = 220;
const DEFAULT_ITEM_WIDTH_EXPANDED = 320;
const DEFAULT_GRID_COLS_EXPANDED = 3;
const DEFAULT_GAP_EXPANDED = 8;

const DEFAULT_HEADER_FOOTER_HEIGHT = 60;
const SIZE_TOLERANCE = 5;

export default function AbstractList({
  data,
  sendStrategyRequest,
  updateEntity,
  viewEntityId,
  parentEntityId,
  entityType,
  listItemViewName,
  expandedViewName,
  defaultItemName,
  initialExpandAllState = false,
  itemHeightCollapsed = DEFAULT_ITEM_HEIGHT_COLLAPSED,
  itemGapCollapsed = DEFAULT_ITEM_GAP_COLLAPSED,
  itemWidthCollapsed = DEFAULT_ITEM_WIDTH_COLLAPSED,
  maxListItemsBeforeScroll = DEFAULT_MAX_LIST_ITEMS_SCROLL,
  itemHeightExpanded = DEFAULT_ITEM_HEIGHT_EXPANDED,
  itemWidthExpanded = DEFAULT_ITEM_WIDTH_EXPANDED,
  gridColsExpanded = DEFAULT_GRID_COLS_EXPANDED,
  gapExpanded = DEFAULT_GAP_EXPANDED,
  headerFooterHeight = DEFAULT_HEADER_FOOTER_HEIGHT,
  enableAutoResize = true, // Auto-resize enabled by default
  hideChildrenInitially = false, // Hide children initially by default
  createMissingViews = true, // Create missing views by default
}: AbstractListProps) {
  const children = useRecoilValue(childrenByTypeSelector({ parentId: parentEntityId, type: entityType })) as any[];
  const [expandAll, setExpandAll] = useState(initialExpandAllState);
  const [hasBeenManuallyResized, setHasBeenManuallyResized] = useState(false);

  const initialResizeDoneRef = useRef(false);
  const lastRequestedSizeRef = useRef<{ width: number; height: number } | null>(null);
  const resizeDebounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const processedChildrenRef = useRef<{ [key: string]: boolean }>({});
  const listContainerRef = useRef<HTMLDivElement>(null);

  // Effect to initially hide children if configured
  useEffect(() => {
    if (!hideChildrenInitially || !children?.length || !updateEntity || !sendStrategyRequest) {
      return;
    }
    const updatesToMake = [];
    children.forEach((child) => {
      if (!processedChildrenRef.current[child.entity_id] && !child.data?.hidden) {
        updatesToMake.push(child);
        // No need to update processedChildrenRef here, it's for one-time processing if needed elsewhere
      }
    });
    if (updatesToMake.length > 0) {
      updatesToMake.forEach((child) => {
        sendStrategyRequest(StrategyRequests.hideEntity(child.entity_id, true));
        updateEntity(child.entity_id, { hidden: true });
        processedChildrenRef.current[child.entity_id] = true; // Mark as processed for hiding
      });
    }
  }, [children, updateEntity, sendStrategyRequest, hideChildrenInitially]);

  // Effect to calculate and request parent resize
  useEffect(() => {
    if (!enableAutoResize || hasBeenManuallyResized || !viewEntityId || !children || children.length === 0) {
      return;
    }

    const isInitialResize = !initialResizeDoneRef.current;
    let desiredWidth: number;
    let desiredHeight: number;
    const numItems = children.length;

    if (expandAll) {
      const cols = gridColsExpanded;
      const rows = Math.ceil(numItems / cols);
      desiredWidth = cols * itemWidthExpanded + (cols > 1 ? (cols - 1) * gapExpanded : 0) + 16; // +16 for p-2 on parent
      desiredHeight = rows * itemHeightExpanded + (rows > 1 ? (rows - 1) * gapExpanded : 0) + headerFooterHeight;
    } else {
      desiredWidth = itemWidthCollapsed;
      const listScrollHeight = listContainerRef.current?.scrollHeight;
      // Buffer for a couple of items to give some space or if items are slightly taller
      const scrollBufferCount = 2; 
      const bufferHeight = scrollBufferCount * (itemHeightCollapsed + itemGapCollapsed);

      if (listScrollHeight && listScrollHeight > 0) {
        // Use scrollHeight if available and greater than a minimal threshold
        desiredHeight = listScrollHeight + headerFooterHeight + bufferHeight;
      } else {
        // Fallback: Estimate height for a certain number of items or max items
        const displayItemsCount = Math.min(numItems, maxListItemsBeforeScroll); 
        const itemsToCalculate = Math.min(numItems + scrollBufferCount, DEFAULT_FALLBACK_MAX_ITEMS);
        desiredHeight = itemsToCalculate * (itemHeightCollapsed + itemGapCollapsed) - (itemsToCalculate > 0 ? itemGapCollapsed : 0) + headerFooterHeight;
      }
    }

    desiredWidth = Math.max(desiredWidth, 250); // Min width
    desiredHeight = Math.max(desiredHeight, 100); // Min height

    const roundedWidth = Math.round(desiredWidth);
    const roundedHeight = Math.round(desiredHeight);

    const lastReq = lastRequestedSizeRef.current;
    if (lastReq && lastReq.width === roundedWidth && lastReq.height === roundedHeight && !isInitialResize) {
      return; 
    }

    lastRequestedSizeRef.current = { width: roundedWidth, height: roundedHeight };
    
    const measurementUsed = !expandAll && listContainerRef.current?.scrollHeight && listContainerRef.current.scrollHeight > 0;
    console.log(`AbstractList: Resizing ${viewEntityId}`, { roundedWidth, roundedHeight, state: expandAll ? 'expanded' : 'collapsed', measured: measurementUsed, numItems });
    sendStrategyRequest(StrategyRequests.setAttributes(viewEntityId, {
      width: roundedWidth,
      height: roundedHeight,
    }, false));

    if (isInitialResize) {
      initialResizeDoneRef.current = true;
    }
  }, [
    children, expandAll, viewEntityId, sendStrategyRequest, hasBeenManuallyResized, enableAutoResize,
    itemHeightCollapsed, itemGapCollapsed, itemWidthCollapsed, maxListItemsBeforeScroll,
    itemHeightExpanded, itemWidthExpanded, gridColsExpanded, gapExpanded, headerFooterHeight
  ]);

  // Effect to detect manual resize
  useEffect(() => {
    if (!enableAutoResize || !initialResizeDoneRef.current || hasBeenManuallyResized || !viewEntityId) return;

    if (resizeDebounceTimerRef.current) {
      clearTimeout(resizeDebounceTimerRef.current);
    }

    resizeDebounceTimerRef.current = setTimeout(() => {
      const currentWidth = data?.width;
      const currentHeight = data?.height;
      const lastRequested = lastRequestedSizeRef.current;

      if (typeof currentWidth === 'number' && typeof currentHeight === 'number' && lastRequested) {
        const widthDiff = Math.abs(currentWidth - lastRequested.width);
        const heightDiff = Math.abs(currentHeight - lastRequested.height);

        if (widthDiff > SIZE_TOLERANCE || heightDiff > SIZE_TOLERANCE) {
          console.log(`AbstractList: Manual resize detected for ${viewEntityId}`, { currentWidth, currentHeight, lastRequested });
          setHasBeenManuallyResized(true);
        }
      }
    }, 250); // Debounce time

    return () => {
      if (resizeDebounceTimerRef.current) {
        clearTimeout(resizeDebounceTimerRef.current);
      }
    };
  }, [data?.width, data?.height, viewEntityId, hasBeenManuallyResized, enableAutoResize]);


  if (!data) {
    return <div>Loading AbstractList or error...</div>;
  }

  return (
    <div className="flex flex-col gap-2 nowheel h-full w-full p-2">
      <div className="flex justify-end gap-2 p-1 flex-shrink-0">
        <button
          onClick={() => {
            setExpandAll(!expandAll);
            // If moving from manual resize back to auto, reset flags
            if(hasBeenManuallyResized) {
                setHasBeenManuallyResized(false);
                initialResizeDoneRef.current = false; // Force recalculation
                lastRequestedSizeRef.current = null; // Clear last requested size
            }
          }}
          className="px-2 py-1 text-xs rounded bg-gray-200 hover:bg-gray-300"
        >
          {expandAll ? 'Collapse All' : 'Expand All'}
        </button>
      </div>

      <div
        ref={listContainerRef}
        className={`overflow-y-auto w-full min-h-0 ${expandAll ? `grid grid-cols-${gridColsExpanded} gap-${Math.round(gapExpanded/4)}` : ''}`} // Tailwind gap is in units of 0.25rem
      >
        {children && children.length > 0 ? children.map((child) => (
          <AbstractListItemRenderer
            key={child.entityId}
            child={child}
            sendStrategyRequest={sendStrategyRequest}
            updateEntity={updateEntity}
            isExpanded={expandAll}
            listItemViewName={listItemViewName}
            expandedViewName={expandedViewName}
            defaultItemName={defaultItemName}
            createMissingViews={createMissingViews}
          />
        )) : (
          <div className="text-center text-gray-500 p-4">No items to display.</div>
        )}
      </div>
    </div>
  );
} 