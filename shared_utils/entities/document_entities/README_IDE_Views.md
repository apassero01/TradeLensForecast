# IDE Views for Document Management

The IDE views provide a complete document management interface that can be added to any entity. The system is self-sufficient - when you create an `ide_app_dashboard` view, it automatically creates the required child views.

## How It Works

The `ide_app_dashboard` view component automatically:
1. Checks if `file_tree` and `document_search` views exist on its parent entity
2. Creates them if they don't exist
3. Renders them in a complete IDE-like interface

The views use React/Recoil selectors to directly query for document children, making them work with any entity that contains DocumentEntity children.

### Document Selection

The IDE uses entity attributes to track which document is selected:
- When a document is selected in the file tree, it sets `ide_selected_by_<parentEntityId>: true` on the document
- The dashboard queries for documents with this attribute to know which one to edit
- Previous selections are cleared automatically
- This approach allows multiple IDE instances to have independent selections

## Usage

### Adding IDE functionality to any entity:

Simply create an `ide_app_dashboard` view on any entity:

```javascript
// Frontend - Add IDE dashboard to an entity
sendStrategyRequest({
  strategy_name: 'CreateEntityStrategy',
  target_entity: someEntityId,
  param_config: {
    entity_class: 'shared_utils.entities.view_entity.ViewEntity.ViewEntity',
    initial_attributes: {
      parent_attributes: {
        "name": "name"
      },
      view_component_type: 'ide_app_dashboard',
      hidden: false
    }
  }
});
```

### Python Backend Example:

```python
# In any entity's on_create method
from shared_utils.entities.view_entity.ViewEntity import ViewEntity
from shared_utils.strategy.BaseStrategy import CreateEntityStrategy

def on_create(self, param_config: Optional[dict] = None) -> list:
    requests = super().on_create(param_config)
    
    # Add IDE dashboard view
    if param_config.get('include_ide', False):
        ide_dashboard_request = CreateEntityStrategy.request_constructor(
            self.entity_id,
            ViewEntity.get_class_path(),
            initial_attributes={
                'parent_attributes': {
                    "name": "name"
                },
                'view_component_type': 'ide_app_dashboard',
                'hidden': False
            }
        )
        requests.append(ide_dashboard_request)
    
    return requests
```

## Components

The IDE system consists of three view components:

1. **`ide_app_dashboard`** - The main dashboard that contains:
   - Header with entity name and document count
   - File tree sidebar
   - Document search
   - Editor area (placeholder for future document editor integration)

2. **`file_tree`** - Automatically created by dashboard if missing
   - Uses `childrenByTypeSelector` to find all DocumentEntity children
   - Shows documents and folders in a tree structure
   - Supports nested folders with recursive document queries
   - Context menus for creating, renaming, deleting

3. **`document_search`** - Automatically created by dashboard if missing
   - Search by name, display name, or content
   - Shows results with file paths
   - Content snippets for text searches

## Benefits

- **Self-sufficient**: The dashboard creates its own dependencies
- **Works with any entity**: No special entity type required
- **Direct queries**: Uses selectors to get document children, no data mapping needed
- **Follows existing patterns**: Similar to MealPlannerDashboard
- **Clean separation**: Views handle presentation, entities handle data

## Troubleshooting

### FileTree not showing documents

If the FileTree is not showing document children:

1. **Check parent entity has document children**: The parent must have DocumentEntity children
2. **Check console logs**: Look for "FileTree documentChildren:" logs to see what's being found
3. **Ensure children are DocumentEntity type**: Only entities with `entity_name = EntityEnum.DOCUMENT` will appear
4. **Check entity relationships**: Ensure documents have correct parent_ids

### Creating test documents

To test the IDE views, create some document children:

```javascript
// Create a test document
sendStrategyRequest({
  strategy_name: 'CreateEntityStrategy',
  target_entity: parentEntityId,
  param_config: {
    entity_class: 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity',
    initial_attributes: {
      name: 'test.py',
      is_folder: false,
      file_type: 'python',
      text: 'print("Hello World")'
    }
  }
});

// Create a folder
sendStrategyRequest({
  strategy_name: 'CreateEntityStrategy',
  target_entity: parentEntityId,
  param_config: {
    entity_class: 'shared_utils.entities.document_entities.DocumentEntity.DocumentEntity',
    initial_attributes: {
      name: 'src',
      is_folder: true,
      file_type: null
    }
  }
});
``` 