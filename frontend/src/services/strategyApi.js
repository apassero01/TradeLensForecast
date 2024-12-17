const API_BASE_URL = 'http://localhost:8000';

class StrategyApi {
  async getHistory() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/get_strategy_history/`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch strategy history');
    }
    return response.json();
  }

  async executeStrategy(strategy) {
    const response = await fetch(`${API_BASE_URL}/training_session/api/execute_strategy/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ strategy })
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to execute strategy');
    }
    return response.json();
  }
}

export const strategyApi = new StrategyApi(); 