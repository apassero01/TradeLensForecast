// src/App.js
import React from 'react';
import { Route, Routes } from 'react-router-dom';   // Import Routes and Route
import TrainingSession from './components/TrainingSession';   // Import the TrainingSession component
import Session from './componentsv2/Session';
import TopLevel from './componentsv2/TopLevel';

function App() {
  return (
    <Routes>  {/* Use Routes instead of Switch */}
      <Route path="/training_session" element={<TrainingSession />} />
      <Route path="/session" element={<Session />} />
      <Route path="/" element={<TopLevel />} />
      {/* You can add other routes here for other parts of your app */}
    </Routes>
  );
}

export default App;