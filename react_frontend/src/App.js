import logo from './logo.svg';
import './App.css';
import TopBar from './components/TopBar/TopBar';
import Canvas from './components/Canvas/Canvas';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';
import RecoilizeDebugger from 'recoilize';
function App() {
  return (
    <RecoilRoot>
      <div className="flex flex-col h-screen overflow-hidden bg-gray-900">
        <div className="w-full">
          <TopBar />
        </div>
        <div className="flex-1 overflow-hidden">
          <Canvas />
        </div>
        <Notification />
      </div>
    </RecoilRoot>
  );
}

export default App;
