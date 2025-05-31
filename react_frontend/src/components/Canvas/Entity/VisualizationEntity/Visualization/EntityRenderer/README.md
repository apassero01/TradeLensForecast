# Entity Extraction and Rendering System

A comprehensive system for extracting, processing, and rendering serialized entity data from various sources like chat responses, API responses, and JSON content.

## Overview

This system provides:
1. **Entity Extraction**: Multiple patterns to detect and parse serialized entity data
2. **View Resolution**: Automatic detection and rendering of entity views
3. **Default Rendering**: Fallback display for entities without views
4. **Chat Integration**: Seamless integration with chat interfaces

## Components

### EntityRenderer

The main component for rendering serialized entity data with intelligent view detection.

```tsx
import EntityRenderer from './EntityRenderer/EntityRenderer';

<EntityRenderer
    entityData={extractedData}
    sendStrategyRequest={sendStrategyRequest}
    updateEntity={updateEntity}
    showBorder={true}
    className="custom-styling"
/>
```

#### Props

- `entityData`: Serialized entity data (single entity or dictionary)
- `sendStrategyRequest`: Function to send strategy requests
- `updateEntity`: Function to update entity state
- `showBorder`: Whether to show borders around rendered entities
- `className`: Additional CSS classes

### useEntityExtractor Hook

A custom hook providing utilities for entity data extraction and processing.

```tsx
import useEntityExtractor from '../../../hooks/useEntityExtractor';

const {
    extractEntityData,
    isValidEntityData,
    processEntityData,
    findViewChildren,
    extractAndProcess
} = useEntityExtractor();
```

#### Methods

- `extractEntityData(content)`: Extract entity data from text content
- `isValidEntityData(data)`: Validate if data is valid entity data
- `processEntityData(rawData)`: Process raw data into standardized format
- `findViewChildren(entity, allEntities)`: Find view children for an entity
- `extractAndProcess(content)`: One-step extraction and processing

## Usage Patterns

### 1. Chat Response Processing

The system automatically detects serialized entity data in chat responses:

```tsx
// In ChatInterface component
const renderMessageContent = (content: string) => {
    const entityData = extractEntityData(content);
    if (entityData) {
        return <EntityRenderer entityData={entityData} />;
    }
    // ... render regular content
};
```

### 2. Manual Entity Rendering

Render entity data directly:

```tsx
const MyComponent = ({ serializedEntity }) => {
    return (
        <EntityRenderer
            entityData={serializedEntity}
            sendStrategyRequest={handleStrategyRequest}
            updateEntity={handleEntityUpdate}
        />
    );
};
```

### 3. Processing API Responses

Extract and process entities from API responses:

```tsx
const { extractAndProcess } = useEntityExtractor();

const handleApiResponse = (response) => {
    const result = extractAndProcess(response.content);
    console.log(`Found ${result.totalEntities} entities`);
    console.log(`Entity types: ${result.entityTypes.join(', ')}`);
    console.log(`Has views: ${result.hasViews}`);
};
```

## Detection Patterns

The system recognizes multiple patterns for entity data:

### 1. Entity Graph Pattern
```
Entity Graph
--------------------------------------------------
{
  "entity_id": "...",
  "entity_type": "...",
  ...
}
==================================================
```

### 2. JSON Code Blocks
```json
{
  "entity_id": "...",
  "entity_type": "...",
  ...
}
```

### 3. StrategyRequest Results
```StrategyRequest
{
  "ret_val": {
    "serialized_entities": { ... }
  }
}
```

### 4. Direct JSON Objects
Any JSON object with `entity_id` and `entity_type` fields.

## View Resolution

The system automatically resolves views for entities:

1. **With Views**: If an entity has view children, renders the first view using the appropriate component
2. **Without Views**: Falls back to a default entity display showing key information

### View Component Mapping

Views are rendered using the `view_component_type` field:

```tsx
// View entity structure
{
  "entity_id": "view-123",
  "entity_type": "view", 
  "view_component_type": "editor", // Maps to Editor component
  "parent_attributes": {
    "data": "content"  // Maps parent.data to view.content
  }
}
```

## Data Flow

1. **Input**: Text content containing serialized entity data
2. **Extraction**: Multiple regex patterns detect and extract JSON data
3. **Validation**: Verify extracted data contains valid entity structure
4. **Processing**: Convert to standardized format and analyze relationships
5. **View Resolution**: Find view children and determine rendering approach
6. **Rendering**: Display using appropriate view component or default display

## Supported Entity Types

The system works with any entity type that follows the standard serialization format:

- `api_model`: API model entities with message history
- `document`: Document entities with text content
- `view`: View entities with component types
- `recipe`: Recipe entities with instructions
- `meal_plan`: Meal planning entities
- Custom entity types following the pattern

## Integration Examples

### Chat Interface
```tsx
// Automatic entity detection in chat messages
const ChatInterface = () => {
    const { extractEntityData } = useEntityExtractor();
    
    const renderMessage = (message) => {
        const entityData = extractEntityData(message.content);
        if (entityData) {
            return <EntityRenderer entityData={entityData} />;
        }
        return <ReactMarkdown>{message.content}</ReactMarkdown>;
    };
};
```

### Custom View Component
```tsx
// Register as a view component type
const visualizationComponents = {
    // ... other components
    entityrenderer: EntityRenderer,
};
```

### Standalone Usage
```tsx
// Use in any component that receives serialized entity data
const EntityDisplay = ({ jsonData }) => {
    const { isValidEntityData } = useEntityExtractor();
    
    if (!isValidEntityData(jsonData)) {
        return <div>Invalid entity data</div>;
    }
    
    return <EntityRenderer entityData={jsonData} />;
};
```

## Backend Integration

When using `serialize_entities_and_strategies: true` in strategy requests, the backend will include entity data in responses that this system can automatically detect and render.

```python
# Backend strategy request
{
    "strategy_name": "CallApiModelStrategy",
    "param_config": {
        "serialize_entities_and_strategies": true,
        "user_input": "Show me the meal plan"
    }
}
```

The response will include serialized entity data that the frontend can automatically extract and render with appropriate views. 