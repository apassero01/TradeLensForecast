import './App.css';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';
import { WebSocketProvider } from './providers/WebSocketProvider';
import ViewSwitcher from './components/ViewSwitcher';

function App() {
  return (
    <RecoilRoot>
      <WebSocketProvider>
        <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-900">
          <ViewSwitcher />
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
