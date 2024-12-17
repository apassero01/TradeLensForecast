export const processEntityGraph = (data, existingNodes = []) => {
  const nodes = [];
  const edges = [];
  let nodeId = 1;

  // Create a map of existing node positions by path
  const existingPositions = new Map(
    existingNodes.map(node => [node.data.path, node.position])
  );

  const processNode = (entity, parentId = null) => {
    // Create node with all entity data
    const node = {
      id: nodeId.toString(),
      type: 'entityNode',
      // Use existing position if available, otherwise calculate new position
      position: existingPositions.get(entity.path) || { x: nodeId * 200, y: nodeId * 100 },
      data: {
        label: entity.entity_name,
        metaData: entity.meta_data || {},
        entity_name: entity.entity_name,
        path: entity.path,
        class_path: entity.class_path,
        visualization: entity.visualization,
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