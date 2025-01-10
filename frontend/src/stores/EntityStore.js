class EntityStore {
    constructor() {
        // Main storage for entities: { entityId: entityData }
        this.entities = new Map();
    }

    // Add or update multiple entities
    updateEntities(entityUpdates) {
        Object.entries(entityUpdates).forEach(([entityId, entityData]) => {
            this.entities.set(entityId, entityData);
        });
    }

    // Get entity by ID
    getEntity(entityId) {
        return this.entities.get(entityId);
    }

    // Convert flat structure to graph format for ReactFlow
    toGraphData() {
        const nodes = [];
        const edges = [];

        this.entities.forEach((entity, entityId) => {
            // Create node
            nodes.push({
                id: entityId,
                type: 'entityNode',
                data: {
                    ...entity,
                    id: entityId,
                },
                position: entity.position || { x: 0, y: 0 } // Position handling will need to be improved
            });

            // Create edges from child relationships
            if (entity.child_ids) {
                entity.child_ids.forEach(childId => {
                    edges.push({
                        id: `${entityId}-${childId}`,
                        source: entityId,
                        target: childId,
                    });
                });
            }
        });

        return { nodes, edges };
    }

    // Get changed nodes and edges based on updated entities
    getChangedGraphData(updatedEntityIds) {
        const changedNodes = [];
        const changedEdges = [];

        updatedEntityIds.forEach(entityId => {
            const entity = this.entities.get(entityId);
            if (!entity) return;

            // Add changed node
            changedNodes.push({
                id: entityId,
                type: 'entityNode',
                data: {
                    ...entity,
                    id: entityId,
                },
                position: entity.position || { x: 0, y: 0 }
            });

            // Add edges for this entity
            if (entity.child_ids) {
                entity.child_ids.forEach(childId => {
                    changedEdges.push({
                        id: `${entityId}-${childId}`,
                        source: entityId,
                        target: childId,
                    });
                });
            }
        });

        return { changedNodes, changedEdges };
    }

    // Clear all entities
    clear() {
        this.entities.clear();
    }
}

export default EntityStore; 