class EntityStore {
    constructor() {
        // Main storage for entities: { entityId: entityData }
        this.entities = new Map();
    }

    // Add or update multiple entities
    updateEntities(entityData) {
        console.log('Updating entities with:', entityData);
        Object.entries(entityData).forEach(([entityId, data]) => {
            this.entities.set(entityId, data);
        });
    }

    // Get entity by ID
    getEntity(entityId) {
        return this.entities.get(entityId);
    }

    // Convert flat structure to graph format for ReactFlow
    toGraphData() {
        const nodes = [];
        const edgesSet = new Set(); // Use Set to track unique edge combinations

        this.entities.forEach((entity, entityId) => {
            console.log('Processing entity for node:', entity);
            
            // Add node
            nodes.push({
                id: entityId,
                type: 'entityNode',
                position: entity.position || { x: 100, y: 100 },
                data: {
                    id: entityId,
                    label: entity.entity_type || 'Entity',
                    path: entity.path,
                    metaData: entity.meta_data || {},
                    visualization: entity.visualization || null,
                    child_ids: entity.child_ids || [],
                    entity_type: entity.entity_type,
                    entity_name: entity.entity_name,
                    ...entity
                }
            });

            // Create edges from child relationships
            if (entity.child_ids) {
                entity.child_ids.forEach(childId => {
                    if (this.entities.has(childId)) {
                        // Create unique edge identifier
                        const edgeId = `edge-${entityId}-${childId}`;
                        
                        // Only add edge if it doesn't already exist
                        if (!edgesSet.has(edgeId)) {
                            edgesSet.add(edgeId);
                            edgesSet.add(`edge-${childId}-${entityId}`); // Prevent reverse edge
                            
                            // Add the edge to our set
                            edgesSet.add({
                                id: edgeId,
                                source: entityId,
                                target: childId,
                                type: 'smoothstep',
                                animated: true,
                            });
                        }
                    }
                });
            }
        });

        console.log('Generated graph data:', { 
            nodes, 
            edges: Array.from(edgesSet).filter(edge => typeof edge === 'object'),
            edgeCount: edgesSet.size 
        });
        
        return { 
            nodes, 
            edges: Array.from(edgesSet).filter(edge => typeof edge === 'object')
        };
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

    removeEntity(entityId) {
        console.log('Removing entity:', entityId);
        if (this.entities.has(entityId)) {
            this.entities.delete(entityId);
        }
    }
}

export default EntityStore; 