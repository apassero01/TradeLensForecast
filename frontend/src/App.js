// src/App.js
import React from 'react';
import { Route, Routes } from 'react-router-dom';   // Import Routes and Route
import TopLevel from './components/TopLevel';

function App() {
  return (
    <Routes>  {/* Use Routes instead of Switch */}
      <Route path="/" element={<TopLevel />} />
      {/* You can add other routes here for other parts of your app */}
    </Routes>
  );
}

export default App;