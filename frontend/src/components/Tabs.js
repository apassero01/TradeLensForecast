import React from 'react';

function Tabs({ activeTab, setActiveTab }) {
  return (
    <div className="flex justify-start items-center bg-gray-100 p-6 rounded-t-lg shadow-sm mb-4">
      <button
        className={`py-2 px-6 text-lg font-semibold ${
          activeTab === 'configuration'
            ? 'border-b-4 border-blue-500 text-blue-600'
            : 'text-gray-600 hover:text-blue-500'
        }`}
        onClick={() => setActiveTab('configuration')}
      >
        Training Configuration
      </button>
      <button
        className={`py-2 px-6 text-lg font-semibold ${
          activeTab === 'preprocessing'
            ? 'border-b-4 border-blue-500 text-blue-600'
            : 'text-gray-600 hover:text-blue-500'
        }`}
        onClick={() => setActiveTab('preprocessing')}
      >
        Preprocessing
      </button>
    </div>
  );
}


export default Tabs;