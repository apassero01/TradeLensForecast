import './App.css';
import TopBar from './components/TopBar/TopBar';
import Canvas from './components/Canvas/Canvas';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';
import { WebSocketProvider } from './providers/WebSocketProvider';

function App() {
  return (
    <RecoilRoot>
      <WebSocketProvider>
        <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-900">
          {/* TopBar with fixed positioning to prevent height changes from affecting layout */}
          <div className="flex-shrink-0">
            <TopBar />
          </div>
          {/* Canvas container that takes remaining space */}
          <div className="flex-1 min-h-0">
            <Canvas />
          </div>
          {/* Notification positioned absolutely to not affect layout */}
          <div className="absolute">
            <Notification />
          </div>
        </div>
      </WebSocketProvider>
    </RecoilRoot>
  );
}

export default App;
