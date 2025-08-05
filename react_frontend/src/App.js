import './App.css';
import { RecoilRoot } from 'recoil';
import { Notification } from './components/Notification/Notification';
import { WebSocketProvider } from './providers/WebSocketProvider';
import ViewSwitcher from './components/ViewSwitcher';
import ErrorBoundary from './components/common/ErrorBoundary';


function App() {
  return (
    <RecoilRoot>
      <ErrorBoundary 
        fallback={(error) => (
          <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
            <div className="max-w-md p-6 bg-red-800 rounded-lg">
              <h2 className="text-xl font-bold mb-2">Application Error</h2>
              <p className="mb-2">Something went wrong:</p>
              <pre className="text-sm bg-gray-800 p-2 rounded overflow-auto">
                {error?.message || error?.toString() || 'Unknown error'}
              </pre>
              <button 
                onClick={() => window.location.reload()} 
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Reload Application
              </button>
            </div>
          </div>
        )}
      >
        <WebSocketProvider>
          <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-900">
            <ViewSwitcher />
            {/* Notification positioned absolutely to not affect layout */}
            <div className="absolute">
              <Notification />
            </div>
          </div>
        </WebSocketProvider>
      </ErrorBoundary>
    </RecoilRoot>
  );
}

export default App;
