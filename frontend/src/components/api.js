import axios from 'axios';

// Fetch sequence sets from the Django backend
export const fetchSequenceSetsMetaData = async () => {
  try {
    const response = await axios.get('http://localhost:8000/sequenceset_manager/get_sequence_metadata/');
    return response.data;
  } catch (err) {
    console.error('Error fetching sequence sets:', err);
    throw err;  // Rethrow the error to handle it in the component
  }
};

export const fetchAllXFeatures = async () => {
  try {
    const response = await axios.get('http://localhost:8000/dataset_manager/get_all_x_features/');
    return response.data;
  } catch (err) {
    console.error('Error fetching X features sets:', err);
    throw err;  // Rethrow the error to handle it in the component
  }
};

export const fetchAllyFeatures = async () => {
  try {
    const response = await axios.get('http://localhost:8000/dataset_manager/get_all_y_features/');
    return response.data;
  } catch (err) {
    console.error('Error fetching y features sets:', err);
    throw err;  // Rethrow the error to handle it in the component
  }
};

export const saveSession = async (sessionState) => {
    try {
        const response = await axios.post('http://localhost:8000/training_session/save_session/', sessionState);
        return response.data;
    } catch (err) {
        console.error('Error saving session:', err);
        throw err;
    }
}

export const removeSession = async (sessionState) => {
    try {
        const response = await axios.post('http://localhost:8000/training_session/remove_session/', sessionState);
        return response.data;
    } catch (err) {
        console.error('Error removing session:', err);
        throw err;
    }
}

export const getSessions = async () => {
    try {
        const response = await axios.get('http://localhost:8000/training_session/get_sessions/');
        return response.data;  // Assuming response data is an array of sessions
    } catch (err) {
        console.error('Error fetching sessions:', err);
        throw err;
    }
};

export const getSessionById = async (sessionId) => {
    try {
        const response = await axios.get(`http://localhost:8000/training_session/get_session/${sessionId}/`);
        return response.data;
    } catch (err) {
        console.error('Error fetching session:', err);
        throw err;
    }
};

export const fetchAvailableEntities = async () => {
  try {
    const response = await axios.get('http://localhost:8000/training_session/api/get_available_entities/');
    return response.data.entities;
  } catch (err) {
    console.error('Error fetching available entities:', err);
    throw err;
  }
};

export const executeStrategy = async (entityId, strategyRequest) => {
  try {
    const response = await axios.post('http://localhost:8000/api/execute_strategy/', {
      entity_id: entityId,
      ...strategyRequest.toJSON()
    });
    return response.data;
  } catch (err) {
    console.error('Error executing strategy:', err);
    throw err;
  }
};
