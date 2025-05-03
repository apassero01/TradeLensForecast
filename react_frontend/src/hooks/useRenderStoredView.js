import React, { useMemo } from 'react';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../state/entitiesSelectors'; // Adjust path as needed
import visualizationComponents from '../components/Canvas/Entity/VisualizationEntity/Visualization/visualizationComponents'; // Adjust path
import ErrorBoundary from '../components/common/ErrorBoundary'; // Adjust path

/**
 * Custom hook to render a view component based on details
 * stored in an entity's Recoil state (`currentView` field).
 *
 * @param {string | null} entityId The ID of the entity whose stored view should be rendered. Can be null/undefined.
 * @returns {React.ReactNode | null} The rendered view component or a fallback/null.
 */
function useRenderStoredView(viewEntityId, sendStrategyRequest, updateEntity) {
    // --- Call hook unconditionally at the top ---
    // Pass entityId directly. Recoil's atomFamily usually provides a default state
    // for unknown/new IDs. If entityId is null/undefined, this might behave unexpectedly
    // depending on atomFamily implementation, but the call itself is unconditional.
    // It's often better practice for the *calling component* to conditionally use the hook:
    // const view = entityId ? useRenderStoredView(entityId) : null; // This is BAD practice itself.
    // Correct calling pattern: const view = useRenderStoredView(entityId); ... if (!entityId) return <Fallback/>
    // We will proceed assuming the hook might receive null/undefined but check *after* the call.
    const viewEntityStore = useRecoilValue(nodeSelectorFamily(viewEntityId));
    const parentEntityStore = useRecoilValue(nodeSelectorFamily(viewEntityStore?.data?.parent_ids?.[0]));

    const viewData = useMemo(() => {
        if (!parentEntityStore || !parentEntityStore.data || !viewEntityStore?.data?.parent_attributes) {
            return {};
        }
        return Object.entries(viewEntityStore.data.parent_attributes).reduce((acc, [parentAttrKey, newKey]) => {
            if (parentEntityStore.data.hasOwnProperty(parentAttrKey)) {
                acc[newKey] = parentEntityStore.data[parentAttrKey];
            }
            return acc;
        }, {});
    }, [parentEntityStore, viewEntityStore?.data?.parent_attributes]);

    const vissualization = {data: viewData}

    // --- Perform checks *after* the unconditional hook call ---

    // 1. Check if a valid entityId was provided initially.
    //    If not, we shouldn't attempt to render anything based on the potentially default state fetched.
    if (!viewEntityId || !parentEntityStore) {
        return null;
    }

    // 2. Check if state was actually found or is valid.
    //    This depends on atomFamily behavior for non-existent IDs. Assume default state might be returned.
    //    A more robust check might be needed if default state looks like valid state.
    if (!viewEntityStore || !parentEntityStore) { // Check if Recoil returned something meaningful
        console.warn(`useRenderStoredView: No valid entity state could be retrieved for ID ${viewEntityId}`);
        return null;
    }

    const view_type = viewEntityStore.data.view_component_type;

    // Handle case where no view details are stored
    if (!viewData || !view_type) {
        // It's valid for an entity to exist but not have view details stored.
        return null;
    }

    // 5. Find the corresponding component constructor
    const ViewComponent = visualizationComponents[view_type];

    // Handle case where the view type is unknown
    if (!ViewComponent) {
        console.error(`useRenderStoredView: No component mapping found for view type "${view_type}"`);
        return (
            <div className="text-red-500 text-sm p-2 border border-red-300 rounded bg-red-50">
                Unknown View Type: {view_type}
            </div>
        );
    }

    // 6. Render the component using the stored props, wrapped in an ErrorBoundary
    return (
        <ErrorBoundary
            fallback={error => (
                <div className="text-red-500 text-sm p-4 border border-red-300 rounded bg-red-50">
                    <div className="font-bold mb-1">Error rendering stored view:</div>
                    <div>{error?.message || "Unknown error"}</div>
                </div>
            )}
        >
            <ViewComponent visualization={vissualization} data={viewData} sendStrategyRequest={sendStrategyRequest} updateEntity={updateEntity} viewEntity={viewEntityStore} />
        </ErrorBoundary>
    );
}

export default useRenderStoredView;