export const processEntityGraph = (data) => {
  const nodes = [];
  const edges = [];
  let nodeId = 1;

  const processNode = (entity, parentId = null) => {
    // Create node with all entity data
    const node = {
      id: nodeId.toString(),
      type: 'entityNode',
      position: { x: nodeId * 200, y: nodeId * 100 }, // Simple positioning logic
      data: {
        label: entity.entity_name,
        metaData: entity.meta_data || {},
        entity_name: entity.entity_name,
        path: entity.path,
        class_path: entity.class_path,
        // Include all other entity data
        ...entity
      }
    };
    
    nodes.push(node);
    
    // Create edge if there's a parent
    if (parentId) {
      edges.push({
        id: `e${parentId}-${nodeId}`,
        source: parentId.toString(),
        target: nodeId.toString(),
        type: 'smoothstep'
      });
    }
    
    const currentId = nodeId;
    nodeId++;
    
    // Process children recursively
    if (entity.children) {
      entity.children.forEach(child => {
        processNode(child, currentId);
      });
    }
  };

  processNode(data);

  return { nodes, edges };
}; 