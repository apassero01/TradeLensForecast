import React, {useEffect, useRef, useState} from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import RecipeListItemRenderer from './RecipeListItemRenderer';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface RecipeListProps {
  data?: RecipeListData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string; 
  parentEntityId: string;
}

interface RecipeListData {
  entity_id: string;
  width?: number;
  height?: number;
}

// Estimated dimensions for calculation (adjust as needed)
const ITEM_HEIGHT_LIST = 48;       // Estimated height (USED AS FALLBACK ONLY)
const ITEM_GAP_LIST = 4;         // Estimated gap (USED AS FALLBACK ONLY)
const ITEM_WIDTH_LIST = 400;       // Target width when collapsed
const MAX_LIST_ITEMS = 40;         // Max items for fallback height calculation

const ITEM_HEIGHT_EXPANDED = 220;  // Estimated height of an item when expanded (adjust)
const ITEM_WIDTH_EXPANDED = 320;   // Increased width for expanded items
const GRID_COLS_EXPANDED = 3;      // Number of columns when expanded
const GAP_EXPANDED = 8;            // Gap size (0.5rem)

const ACCURATE_HEADER_FOOTER_HEIGHT = 60; // More precise estimate: p-2(8+8) + p-1(4+4) + button(~24) + gap-2(8) = 56 -> 60
const SIZE_TOLERANCE = 5;

export default function RecipeList({ data, sendStrategyRequest, updateEntity, viewEntityId, parentEntityId }: RecipeListProps) {
  const recipeChildren = useRecoilValue(childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.RECIPE })) as any[];
  const [expandAll, setExpandAll] = useState(false);
  const [hasBeenManuallyResized, setHasBeenManuallyResized] = useState(false);

  // Refs
  const initialResizeDoneRef = useRef(false);
  const lastRequestedSizeRef = useRef<{width: number; height: number} | null>(null);
  const resizeDebounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const processedChildrenRef = useRef({});
  const listContainerRef = useRef<HTMLDivElement>(null); // Ref for the list container

  useEffect(() => {
    if (recipeChildren?.length > 0) {
      const updatesToMake = [];
      recipeChildren.forEach((child) => {
        if (!processedChildrenRef.current[child.entity_id]) {
          updatesToMake.push(child);
          processedChildrenRef.current[child.entity_id] = true;
        }
      });
      if (updatesToMake.length > 0 && updateEntity && sendStrategyRequest) {
         updatesToMake.forEach((child) => {
            if (!child.data?.hidden) {
                sendStrategyRequest(StrategyRequests.hideEntity(child.entity_id, true));
                updateEntity(child.entity_id, { hidden: true });
            }
         });
      }
    }
  }, [recipeChildren, updateEntity, sendStrategyRequest]);

  // Effect to calculate and request parent resize (INITIAL or on EXPAND ALL change)
  useEffect(() => {
    if (hasBeenManuallyResized || !viewEntityId || recipeChildren.length === 0) {
        return;
    }

    const isInitialResize = !initialResizeDoneRef.current;
    let desiredWidth;
    let desiredHeight;
    const numItems = recipeChildren.length;

    if (expandAll) {
      // ---- Expanded State (Grid Calculation) ----
      const cols = GRID_COLS_EXPANDED;
      const rows = Math.ceil(numItems / cols);
      desiredWidth = cols * ITEM_WIDTH_EXPANDED + (cols - 1) * GAP_EXPANDED + 16;
      desiredHeight = rows * ITEM_HEIGHT_EXPANDED + (rows - 1) * GAP_EXPANDED + ACCURATE_HEADER_FOOTER_HEIGHT;
    } else {
      // ---- Collapsed State (Measure Scroll Height + Buffer) ----
      desiredWidth = ITEM_WIDTH_LIST;
      const listScrollHeight = listContainerRef.current?.scrollHeight;
      const twoItemBuffer = 2 * (ITEM_HEIGHT_LIST + ITEM_GAP_LIST);

      if (listScrollHeight && listScrollHeight > 0) {
        // If measurable, use scroll height + header/footer + buffer for 2 items
        desiredHeight = listScrollHeight + ACCURATE_HEADER_FOOTER_HEIGHT + twoItemBuffer;
      } else {
        // Fallback: Estimate height for (items + 2), up to MAX_LIST_ITEMS
        const displayItemsCount = Math.min(numItems + 2, MAX_LIST_ITEMS);
        desiredHeight = displayItemsCount * (ITEM_HEIGHT_LIST + ITEM_GAP_LIST) + ACCURATE_HEADER_FOOTER_HEIGHT;
      }
    }

    // Apply minimum dimensions
    desiredWidth = Math.max(desiredWidth, 250);
    desiredHeight = Math.max(desiredHeight, 100);

    const roundedWidth = Math.round(desiredWidth);
    const roundedHeight = Math.round(desiredHeight);

    const lastReq = lastRequestedSizeRef.current;
    if (lastReq && lastReq.width === roundedWidth && lastReq.height === roundedHeight) {
        return; // No change needed
    }

    lastRequestedSizeRef.current = { width: roundedWidth, height: roundedHeight };

    // Determine if measurement was used for logging
    const measurementUsed = !expandAll && listContainerRef.current?.scrollHeight && listContainerRef.current.scrollHeight > 0;

    console.log("Sending request (measure/estimate)", { viewEntityId, roundedWidth, roundedHeight, state: expandAll ? 'expanded' : 'collapsed', measured: measurementUsed });
    sendStrategyRequest(StrategyRequests.setAttributes(viewEntityId, {
       width: roundedWidth,
       height: roundedHeight,
    }, false));

    if (isInitialResize) {
        initialResizeDoneRef.current = true;
    }

  }, [recipeChildren, expandAll, viewEntityId, sendStrategyRequest, hasBeenManuallyResized]);

  // Effect to DETECT manual resize (DEBOUNCED)
  useEffect(() => {
    // Clear any existing timer
    if (resizeDebounceTimerRef.current) {
      clearTimeout(resizeDebounceTimerRef.current);
    }

    // Don't check until initial resize is done, and if already marked as manual
    if (!initialResizeDoneRef.current || hasBeenManuallyResized || !viewEntityId) return;

    resizeDebounceTimerRef.current = setTimeout(() => {
      const currentWidth = data?.width;
      const currentHeight = data?.height;
      const lastRequested = lastRequestedSizeRef.current;

      if (typeof currentWidth === 'number' && typeof currentHeight === 'number' && lastRequested) {
          const widthDiff = Math.abs(currentWidth - lastRequested.width);
          const heightDiff = Math.abs(currentHeight - lastRequested.height);

          if (widthDiff > SIZE_TOLERANCE || heightDiff > SIZE_TOLERANCE) {
              console.log("Manual resize detected (debounced)", { currentWidth, currentHeight, lastRequested });
              setHasBeenManuallyResized(true);
          }
      }
    }, 250);

    return () => {
        if (resizeDebounceTimerRef.current) {
            clearTimeout(resizeDebounceTimerRef.current);
        }
    };
  }, [data?.width, data?.height, viewEntityId, hasBeenManuallyResized]);

  if (!data) {
    return <div>Loading or error...</div>;
  }

  return (
    <div className="flex flex-col gap-2 nowheel h-full w-full p-2">
      {/* Expand/Collapse Button */}
      <div className="flex justify-end gap-2 p-1 flex-shrink-0">
         <button
           onClick={() => setExpandAll(!expandAll)}
           className="px-2 py-1 text-xs rounded bg-gray-200 hover:bg-gray-300"
         >
           {expandAll ? 'Collapse All' : 'Expand All'}
         </button>
      </div>

      {/* Main Content Area - Attach ref */}
      <div
        ref={listContainerRef} // Attach the ref here
        className={`overflow-y-auto w-full min-h-0 ${expandAll ? 'grid grid-cols-3 gap-2' : ''}`}>
        {recipeChildren.map((child) => (
          <RecipeListItemRenderer
            key={child.entityId}
            child={child}
            sendStrategyRequest={sendStrategyRequest}
            updateEntity={updateEntity}
            isExpanded={expandAll}
          />
        ))}
      </div>
    </div>
  );
}