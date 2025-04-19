# Strategy Request Builder

The Strategy Request Builder is a utility for simplifying the creation of strategy requests in the frontend application. It provides a consistent structure for building strategy request objects that are sent to the backend through WebSocket connections.

## Basic Usage

Import the StrategyRequests utility:

```javascript
import StrategyRequests from '../utils/StrategyRequestBuilder';
```

### Using Shorthand Methods

For common strategy request types, you can use the built-in factory methods:

```javascript
// Execute all strategy request children of an entity
const request = StrategyRequests.executeChildren(entityId);
sendStrategyRequest(request);

// Set attributes on an entity
const attributesRequest = StrategyRequests.setAttribute(
  entityId, 
  { user_input: "New text" },
  false  // addToHistory
);
sendStrategyRequest(attributesRequest);

// Update children order
const orderRequest = StrategyRequests.updateChildrenOrder(
  parentId,
  [childId1, childId2, childId3]
);
sendStrategyRequest(orderRequest);

// Create a new entity
const createRequest = StrategyRequests.createEntity(
  parentId,
  'view',
  { view_component_type: 'editor' }
);
sendStrategyRequest(createRequest);
```

### Using the Builder Pattern

For more complex requests, you can use the builder pattern:

```javascript
const complexRequest = StrategyRequests.builder()
  .withStrategyName('CustomStrategy')
  .withTargetEntity(entityId)
  .withParam('key1', 'value1')
  .withParam('key2', 'value2')
  .withAddToHistory(true)
  .withNestedRequest(StrategyRequests.setAttribute(childId, { status: 'complete' }))
  .build();

sendStrategyRequest(complexRequest);
```

## Strategy Request Structure

Each strategy request follows this structure:

```javascript
{
  strategy_name: string,       // Name of the strategy to execute
  target_entity_id: string,    // ID of the entity to target
  param_config: object,        // Parameters for the strategy
  add_to_history: boolean,     // Whether to add this to history
  nested_requests: array       // Nested strategy requests to execute
}
```

## Available Builder Methods

The builder provides these methods for constructing requests:

- `withStrategyName(name)` - Set the strategy name
- `withTargetEntity(entityId)` - Set the target entity ID
- `withParams(paramsObject)` - Add multiple parameters at once
- `withParam(key, value)` - Add a single parameter
- `withAddToHistory(boolean)` - Set whether to add to history
- `withNestedRequest(requestObject)` - Add a nested request
- `withNestedRequests(requestsArray)` - Add multiple nested requests
- `build()` - Build and return the final request object
- `reset()` - Reset the builder to its initial state

## Common Strategy Types

Here are some common strategy types used in the application:

- `SetAttributesStrategy` - Set attributes on an entity
- `UpdateChildrenStrategy` - Update child entities order
- `ExecuteRequestChildren` - Execute all strategy request children
- `CreateEntityStrategy` - Create a new entity