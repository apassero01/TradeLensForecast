import React from 'react';

function Tabs({ activeTab, setActiveTab }) {
  return (
      <div
          className="flex justify-start items-center bg-[#2b2b2b] p-4 rounded-t-lg shadow-lg mb-0 border-b border-gray-700">
          {/* Configuration Tab */}
          <button
              className={`py-2 px-4 text-md font-semibold transition-colors duration-300 ${
                  activeTab === 'configuration'
                      ? 'border-b-4 border-blue-500 text-blue-500'
                      : 'text-gray-400 hover:text-blue-400'
              }`}
              onClick={() => setActiveTab('configuration')}
          >
              Training Configuration
          </button>

          {/* Preprocessing Tab */}
          <button
              className={`py-2 px-4 text-md font-semibold transition-colors duration-300 ${
                  activeTab === 'preprocessing'
                      ? 'border-b-4 border-blue-500 text-blue-500'
                      : 'text-gray-400 hover:text-blue-400'
              }`}
              onClick={() => setActiveTab('preprocessing')}
          >
              Preprocessing
          </button>
          <button
              className={`py-2 px-4 text-md font-semibold transition-colors duration-300 ${
                  activeTab === 'ModelStage'
                      ? 'border-b-4 border-blue-500 text-blue-500'
                      : 'text-gray-400 hover:text-blue-400'
              }`}
              onClick={() => setActiveTab('ModelStage')}
          >
              Model Stage
          </button>
      </div>
  );
}

export default Tabs;