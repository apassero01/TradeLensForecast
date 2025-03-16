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
      <div className="flex flex-col h-screen overflow-hidden bg-gray-900">
        <div className="w-full">
          <TopBar />
        </div>
        <div className="flex-1 overflow-hidden">
          <Canvas />
        </div>
          <Notification />
        </div>
      </WebSocketProvider>
    </RecoilRoot>
  );
}

export default App;
