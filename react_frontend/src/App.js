import logo from './logo.svg';
import './App.css';
import TopBar from './components/TopBar/TopBar';
import Canvas from './components/Canvas/Canvas';
import StrategyManager from './components/Strategy/StrategyManager';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';

function App() {
  return (
    <RecoilRoot>
      <div className="min-h-screen bg-gray-900">
        <TopBar />
        <div className="flex flex-col items-center justify-center h-screen">
          <Canvas />
          <StrategyManager />
        </div>
        <Notification />
      </div>
    </RecoilRoot>
  );
}

export default App;
