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

