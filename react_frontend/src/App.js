import './App.css';
import TopBar from './components/TopBar/TopBar';
import Canvas from './components/Canvas/Canvas';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';
import { WebSocketProvider } from './providers/WebSocketProvider';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import EntityViewPage from './components/EntityViewPage/EntityViewPage';

function App() {
  return (
    <RecoilRoot>
      <WebSocketProvider>
        <Router>
          <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-900">
            {/* Main content area with routing */}
            <Routes>
              <Route path="/" element={
                <>
                  {/* TopBar only shown on canvas route */}
                  <div className="flex-shrink-0">
                    <TopBar />
                  </div>
                  <div className="flex-1 min-h-0">
                    <Canvas />
                  </div>
                </>
              } />
              <Route path="/entity/:entityId" element={<EntityViewPage />} />
            </Routes>
            {/* Notification positioned absolutely to not affect layout */}
            <div className="absolute">
              <Notification />
            </div>
          </div>
        </Router>
      </WebSocketProvider>
    </RecoilRoot>
  );
}

export default App;
