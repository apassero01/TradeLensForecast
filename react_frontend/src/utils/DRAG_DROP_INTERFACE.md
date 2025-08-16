# Standardized Drag & Drop Interface for Entity Management

## Overview

This document describes the standardized drag-and-drop interface for moving entities across all components in the application. Every component that handles entity dragging or dropping should use these utilities to ensure consistency and interoperability.

## Core Principles

1. **Universal Entity ID**: Every drag operation transfers the entity ID as the primary data
2. **Consistent Data Format**: Standardized data structure for all drag operations
3. **Source Context Tracking**: Know where entities are being dragged from
4. **Visual Feedback**: Consistent drag images and drop zone indicators
5. **Type Safety**: TypeScript interfaces for all drag/drop operations

## Usage

### Import the Utilities

```typescript
import { 
  EntityDragDropUtil, 
  EntityDragData, 
  useDragDrop,
  makeDraggable 
} from '../utils/dragDropInterface';
```

### Making Components Draggable

#### Basic Entity Dragging

```typescript
const handleDragStart = (e: React.DragEvent) => {
  EntityDragDropUtil.startDrag(e, {
    entityId: 'entity-123',
    entityType: 'document',
    sourceContext: 'file-tree',
    sourceParentId: 'parent-456',
  }, {
    dragEffect: 'move',
    dragImageText: '📄 My Document',
  });
};

return (
  <div
    draggable
    onDragStart={handleDragStart}
    className="cursor-grab"
  >
    My Entity Content
  </div>
);
```

#### Using the HOC for Auto-Draggable Components

```typescript
const MyEntityComponent = ({ entityId, name, type }) => (
  <div>{name}</div>
);

const DraggableEntity = makeDraggable(MyEntityComponent, {
  getDragData: (props) => ({
    entityId: props.entityId,
    entityType: props.type,
    sourceContext: 'my-component',
  }),
  dragImageText: (props) => `${props.type} ${props.name}`,
});
```

### Creating Drop Zones

#### Using the useDragDrop Hook

```typescript
const MyDropZone = () => {
  const dropHandler = useDragDrop({
    dropEffect: 'move',
    onDrop: (data: EntityDragData) => {
      console.log('Dropped entity:', data.entityId);
      // Handle the drop (e.g., move entity to new parent)
      handleEntityMove(data.entityId, newParentId);
    },
    onDragOver: (event) => {
      // Optional: Add visual feedback
    },
  });

  return (
    <div
      className="drop-zone border-dashed border-2"
      {...dropHandler}
    >
      Drop entities here
    </div>
  );
};
```

#### Manual Drop Handling

```typescript
const handleDrop = (e: React.DragEvent) => {
  EntityDragDropUtil.handleDrop(e, (data: EntityDragData) => {
    // Process the dropped entity
    moveEntity(data.entityId, targetLocation);
  });
};

const handleDragOver = (e: React.DragEvent) => {
  EntityDragDropUtil.handleDragOver(e, {
    dropEffect: 'move',
  });
};
```

## Data Structure

### EntityDragData Interface

```typescript
interface EntityDragData {
  entityId: string;           // Required: The entity being dragged
  entityType?: string;        // Optional: Type of entity
  sourceContext?: string;     // Optional: Where it came from
  sourceParentId?: string;    // Optional: Original parent
  customData?: Record<string, any>; // Optional: Additional data
}
```

## Standard Source Contexts

Use these standardized source context identifiers:

- `'entity-explorer'` - Main entity tree
- `'file-tree'` - File tree components
- `'agent-dashboard-explorer'` - Agent dashboard entity explorer
- `'agent-dashboard-context'` - Agent dashboard context pills
- `'canvas'` - Main canvas entities
- `'search-results'` - Search result items
- `'recipe-list'` - Recipe listing
- `'meal-planner'` - Meal planning interface

## Examples by Component Type

### Entity Explorer Trees

```typescript
// In EntityExplorer, FileTree, etc.
const EntityTreeItem = ({ entityId }) => {
  const handleDragStart = (e: React.DragEvent) => {
    EntityDragDropUtil.startDrag(e, {
      entityId,
      entityType: node.data.entity_type,
      sourceContext: 'entity-explorer',
      sourceParentId: node.data.parent_ids?.[0],
    }, {
      dragImageText: `${getIcon(type)} ${displayName}`,
    });
  };

  return (
    <div draggable onDragStart={handleDragStart}>
      {/* Entity content */}
    </div>
  );
};
```

### Context Management Areas

```typescript
// Pinned context, working context, etc.
const ContextArea = () => {
  const dropHandler = useDragDrop({
    onDrop: (data: EntityDragData) => {
      addToContext(data.entityId);
    },
  });

  return (
    <div {...dropHandler} className="context-drop-zone">
      {/* Context pills */}
    </div>
  );
};
```

### Cross-Component Transfers

```typescript
// Moving from File Tree to Agent Context
const handleAgentContextDrop = (data: EntityDragData) => {
  if (data.sourceContext === 'file-tree') {
    // Moving from file tree to agent context
    sendStrategyRequest(
      StrategyRequests.builder()
        .withStrategyName('SetAttributesStrategy')
        .withTargetEntity(agentId)
        .withParams({
          attribute_map: {
            pinned_entity_ids: [...pinnedIds, data.entityId]
          }
        })
        .build()
    );
  }
};
```

## Visual Feedback

### Drag Images

The interface automatically creates custom drag images with:
- Entity icon + name
- Consistent styling
- Proper cleanup

### Drop Zone Indicators

Add visual feedback in your drop zones:

```css
.drop-zone:hover {
  border-color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.1);
}

.drop-zone.drag-over {
  border-color: #10b981;
  background-color: rgba(16, 185, 129, 0.1);
}
```

## Best Practices

1. **Always use EntityDragDropUtil**: Don't implement custom drag/drop logic
2. **Include source context**: Always specify where the drag originated
3. **Provide meaningful drag images**: Use entity icons and names
4. **Handle edge cases**: Check for valid entity IDs and permissions
5. **Log operations**: Use console.log for debugging drag/drop operations
6. **Test cross-component**: Ensure entities can move between different UI areas

## Migration Guide

### From Old Drag/Drop Code

**Before:**
```typescript
const handleDragStart = (e: React.DragEvent) => {
  e.dataTransfer.setData('application/x-entity-id', entityId);
  e.dataTransfer.setData('text/plain', entityId);
};

const handleDrop = (e: React.DragEvent) => {
  e.preventDefault();
  const entityId = e.dataTransfer.getData('application/x-entity-id');
  // Handle drop
};
```

**After:**
```typescript
const handleDragStart = (e: React.DragEvent) => {
  EntityDragDropUtil.startDrag(e, {
    entityId,
    sourceContext: 'my-component',
  });
};

const dropHandler = useDragDrop({
  onDrop: (data: EntityDragData) => {
    // Handle drop with full context
  },
});
```

## Debugging

Enable drag/drop logging:
```typescript
// The utilities automatically log drag start/drop operations
// Look for these in the console:
// 🎯 Entity drag started: { entityId: "...", sourceContext: "..." }
// 🎯 Entity drop handled: { entityId: "...", sourceContext: "..." }
```

## Future Enhancements

- Batch drag/drop for multiple entities
- Drag/drop permissions based on entity types
- Visual drag previews for complex operations
- Undo/redo for drag/drop operations
- Cross-window drag/drop support


