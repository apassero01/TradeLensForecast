// Central place for all API calls
const API_BASE_URL = 'http://localhost:8000';

class EntityApi {
  async startSession() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/start_session/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to start session');
    }
    return response.json();
  }

  async getEntityGraph() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/get_entity_graph/`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch graph data');
    }
    console.log('getEntityGraph response', response)
    return response.json();
  }

  async stopSession() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/stop_session/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to stop session');
    }
    return response.json();
  }

  async saveSession() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/save_session/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to save session');
    }
    return response.json();
  }

  async getSavedSessions() {
    console.log('Calling getSavedSessions API...');
    const response = await fetch(`${API_BASE_URL}/training_session/api/get_saved_sessions/`);
    console.log('Raw response:', response);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch saved sessions');
    }
    const data = await response.json();
    console.log('Parsed response:', data);
    return data;
  }

  async loadSession(sessionId) {
    console.log('Loading session:', sessionId);
    const response = await fetch(`${API_BASE_URL}/training_session/api/load_session/${sessionId}/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to load session');
    }
    return response.json();
  }

  async getStrategyRegistry() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/get_strategy_registry/`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch strategy registry');
    }
    return response.json();
  }

  async fetchAvailableEntities() {
    const response = await fetch(`${API_BASE_URL}/training_session/api/get_available_entities/`);
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to fetch available entities');
    }
    const data = await response.json();
    return data.entities;
  }
}
    
export const entityApi = new EntityApi(); 