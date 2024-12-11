const VERTICAL_SPACING = 150;  // Space below parent
const HORIZONTAL_SPACING = 250; // Space between siblings

export const calculateNodeHeight = (node) => {
  const baseHeight = 100;
  const contentLines = node.data?.metaData ? Object.keys(node.data.metaData).length : 0;
  return baseHeight + (contentLines * 20);
};

export const calculateNewNodePosition = (parentNode, siblings) => {
  if (!parentNode) return { x: 0, y: 0 };

  const parentY = parentNode.position.y;
  const parentX = parentNode.position.x;
  const parentHeight = calculateNodeHeight(parentNode);
  const newY = parentY + parentHeight + VERTICAL_SPACING;

  // If no siblings, place directly under parent
  if (siblings.length === 0) {
    return { x: parentX, y: newY };
  }

  // Get the last sibling's position
  const lastSibling = siblings[siblings.length - 1];
  const newX = lastSibling.position.x + HORIZONTAL_SPACING;

  return { x: newX, y: newY };
}; 