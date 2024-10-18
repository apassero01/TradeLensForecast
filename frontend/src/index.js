// src/index.js
import React from 'react';
import { createRoot } from 'react-dom/client';   // Import createRoot from react-dom/client
import App from './App';
import { BrowserRouter } from 'react-router-dom';

// Find the root element in the HTML
const container = document.getElementById('root');
const root = createRoot(container);  // Create a root for React to manage

// Render the app wrapped in BrowserRouter and StrictMode
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
