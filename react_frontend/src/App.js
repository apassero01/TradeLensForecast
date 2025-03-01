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

      <div className="min-h-screen bg-gray-900">
        <TopBar />
        <div className="flex flex-col items-center justify-center h-screen">
          <Canvas />
        </div>
          <Notification />
        </div>
        {/* <RecoilizeDebugger /> */}
    </RecoilRoot>
  );
}

export default App;
