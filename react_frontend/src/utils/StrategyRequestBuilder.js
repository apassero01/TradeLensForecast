/**
 * StrategyRequestBuilder - A utility class for constructing strategy requests
 * 
 * This builder provides a fluent interface for creating strategy request objects
 * that can be sent through the WebSocket connection.
 */

class StrategyRequestBuilder {
  constructor() {
    this.reset();
  }

  /**
   * Reset the builder to its initial state
   */
  reset() {
    this.request = {
      strategy_name: '',
      target_entity_id: null,
      param_config: {},
      add_to_history: true,
      nested_requests: [],
    };
    return this;
  }

  /**
   * Set the strategy name
   * @param {string} name - Name of the strategy to execute
   */
  withStrategyName(name) {
    this.request.strategy_name = name;
    return this;
  }

  /**
   * Set the target entity ID
   * @param {string|number} entityId - ID of the entity to target
   */
  withTargetEntity(entityId) {
    this.request.target_entity_id = entityId;
    return this;
  }

  /**
   * Add parameters to the param_config object
   * @param {Object} params - Key-value pairs to add to param_config
   */
  withParams(params) {
    this.request.param_config = {
      ...this.request.param_config,
      ...params
    };
    return this;
  }

  /**
   * Set a single parameter value
   * @param {string} key - Parameter name
   * @param {any} value - Parameter value
   */
  withParam(key, value) {
    this.request.param_config[key] = value;
    return this;
  }

  /**
   * Set whether to add this request to history
   * @param {boolean} addToHistory
   */
  withAddToHistory(addToHistory) {
    this.request.add_to_history = addToHistory;
    return this;
  }

  /**
   * Add a nested request
   * @param {Object} nestedRequest - A strategy request object
   */
  withNestedRequest(nestedRequest) {
    this.request.nested_requests.push(nestedRequest);
    return this;
  }

  /**
   * Add multiple nested requests
   * @param {Array} nestedRequests - Array of strategy request objects
   */
  withNestedRequests(nestedRequests) {
    this.request.nested_requests = [
      ...this.request.nested_requests,
      ...nestedRequests
    ];
    return this;
  }

  /**
   * Build and return the strategy request object
   */
  build() {
    const result = { ...this.request };
    // Return a copy to prevent mutation of the built object
    return result;
  }
}

/**
 * Factory functions for common strategy request types
 */
export const StrategyRequests = {
  /**
   * Create a builder instance
   */
  builder() {
    return new StrategyRequestBuilder();
  },

  /**
   * Set attributes on an entity
   * @param {string|number} entityId - Target entity ID
   * @param {Object} attributes - Key-value pairs of attributes to set
   * @param {boolean} addToHistory - Whether to add to history (default: false)
   */
  setAttributes(entityId, attributes, addToHistory = false) {
    return new StrategyRequestBuilder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(entityId)
      .withParams({ attribute_map: attributes })
      .withAddToHistory(addToHistory)
      .build();
  },

//   /**
//    * Update children order for an entity
//    * @param {string|number} entityId - Parent entity ID
//    * @param {Array} childIds - Ordered array of child entity IDs
//    */
//   updateChildrenOrder(entityId, childIds) {
//     return new StrategyRequestBuilder()
//       .withStrategyName('UpdateChildrenStrategy')
//       .withTargetEntity(entityId)
//       .withParams({ child_ids: childIds })
//       .withAddToHistory(false)
//       .build();
//   },

  /**
   * Execute all child strategy requests of an entity
   * @param {string|number} entityId - Entity ID
   */
  executeChildren(entityId) {
    return new StrategyRequestBuilder()
      .withStrategyName('ExecuteRequestChildren')
      .withTargetEntity(entityId)
      .withAddToHistory(false)
      .build();
  },

  /**
   * Request to create a new entity
   * @param {string|number} parentEntityId - Parent entity ID
   * @param {string} entityType - Type of entity to create
   * @param {Object} initialAttributes - Initial attributes for the new entity
   */
  createEntity(parentEntityId, entityType, initialAttributes = {}) {
    return new StrategyRequestBuilder()
      .withStrategyName('CreateEntityStrategy')
      .withTargetEntity(parentEntityId)
      .withParams({
        entity_class: entityType,
        initial_attributes: initialAttributes
      })
      .withAddToHistory(false)
      .build();
  },

  hideEntity(entityId, hide = true) {
    return StrategyRequests.setAttributes(entityId, { hidden: hide });
  }
};


export default StrategyRequests; 