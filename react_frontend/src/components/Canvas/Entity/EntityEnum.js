// enums.js

// Define entity types
export const EntityTypes = Object.freeze({
    STRATEGY_REQUEST: 'strategy_request',
    INPUT: 'input',
    ENTITY: 'entity',
    VISUALIZATION: 'visualization',
    DOCUMENT: 'document',
    VIEW: 'view',
  });
  
  // Define corresponding React Flow node types
export const NodeTypes = Object.freeze({
    STRATEGY_REQUEST_ENTITY: 'strategyRequestEntity',
    ENTITY_NODE: 'entityNode',
    INPUT_ENTITY: 'inputEntity',
    VISUALIZATION_ENTITY: 'visualizationEntity',
    DOCUMENT_ENTITY: 'documentEntity',
    VIEW_ENTITY: 'viewEntity',
  });