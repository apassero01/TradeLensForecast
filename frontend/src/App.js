// src/App.js
import React from 'react';
import { Route, Routes } from 'react-router-dom';   // Import Routes and Route
import TrainingSession from './components/TrainingSession';   // Import the TrainingSession component

function App() {
  return (
    <Routes>  {/* Use Routes instead of Switch */}
      <Route path="/training_session" element={<TrainingSession />} />
      {/* You can add other routes here for other parts of your app */}
    </Routes>
  );
}

export default App;