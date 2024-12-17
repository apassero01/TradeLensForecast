class StrategyRequest {
  constructor(strategy) {
    this.strategy_name = strategy.name;
    this.strategy_path = strategy.path;
    this.param_config = strategy.config?.param_config || strategy.config || {};
    this.nested_requests = strategy.nested_requests || [];
    this.add_to_history = strategy.add_to_history || false;
  }

  // Update a specific parameter in param_config
  setParameter(key, value) {
    this.param_config[key] = value;
  }

  // Get the request object in the format the backend expects
  toJSON() {
    return {
      strategy_name: this.strategy_name,
      strategy_path: this.strategy_path,
      param_config: this.param_config,
      nested_requests: this.nested_requests,
      add_to_history: this.add_to_history
    };
  }
}

export default StrategyRequest; 