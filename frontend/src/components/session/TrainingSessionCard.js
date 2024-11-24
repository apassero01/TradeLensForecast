// TrainingSessionCard.js

import React, { useState } from 'react';

function TrainingSessionCard({ sessionState }) {
  const sessionData = sessionState?.sessionData;
  const [isModelSetsOpen, setModelSetsOpen] = useState(false);

  const toggleModelSets = () => {
    setModelSetsOpen(!isModelSetsOpen);
  };

  const formatShape = (shape) => {
    return shape ? `(${shape.join(', ')})` : 'None';
  };


  return (
    <div className="bg-[#1e1e1e] shadow-lg rounded-lg p-6 w-full max-w-md border border-gray-700 text-gray-200">
      {!sessionData ? (
        <p className="text-gray-400">Loading session data...</p>
      ) : (
        <>
          {/* Session Header */}
          <div className="mb-4">
            <h2 className="text-2xl font-bold text-blue-500 mb-1">
              Session ID: <span className="text-gray-100">{sessionData.session_id}</span>
            </h2>
            <p className="text-sm text-gray-400">
              Status:{' '}
              <span className="font-mono text-gray-300">{sessionData.status}</span>
            </p>
            <p className="text-sm text-gray-400">
              Created At:{' '}
              <span className="font-mono text-gray-300">
                {new Date(sessionData.created_at).toLocaleString()}
              </span>
            </p>
            <p className="text-sm text-gray-400">
              Start Date:{' '}
              <span className="font-mono text-gray-300">
                {new Date(sessionData.start_date).toLocaleDateString()}
              </span>
            </p>
            <p className="text-sm text-gray-400">
              End Date:{' '}
              <span className="font-mono text-gray-300">
                {sessionData.end_date
                  ? new Date(sessionData.end_date).toLocaleDateString()
                  : 'N/A'}
              </span>
            </p>
          </div>

          {/* Features Section */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-blue-400 mb-2">Features</h3>
            <div className="grid grid-cols-1 gap-2">
              {/* X_features */}
              <div>
                <p className="text-sm text-gray-400 mb-1">X_features:</p>
                <div className="bg-[#2b2b2b] p-2 rounded border border-gray-700">
                  <div className="flex flex-wrap font-mono text-xs text-gray-300">
                    {sessionData.X_features?.map((feature, index) => (
                      <span key={index} className="mr-2">
                        "{feature}",
                      </span>
                    )) || 'N/A'}
                  </div>
                </div>
              </div>
              {/* y_features */}
              <div>
                <p className="text-sm text-gray-400 mb-1">y_features:</p>
                <div className="bg-[#2b2b2b] p-2 rounded border border-gray-700">
                  <div className="flex flex-wrap font-mono text-xs text-gray-300">
                    {sessionData.y_features?.map((feature, index) => (
                      <span key={index} className="mr-2">
                        "{feature}",
                      </span>
                    )) || 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Data Shapes */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-blue-400 mb-2">Data Shapes</h3>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm font-mono text-gray-300">
              <p>X_train:</p>
              <p className="text-right">{formatShape(sessionData.X_train)}</p>
              <p>y_train:</p>
              <p className="text-right">{formatShape(sessionData.y_train)}</p>
              <p>X_test:</p>
              <p className="text-right">{formatShape(sessionData.X_test)}</p>
              <p>y_test:</p>
              <p className="text-right">{formatShape(sessionData.y_test)}</p>
            </div>
          </div>

          {/* Model Sets Section */}
          <div className="mb-4">
            <button
              onClick={toggleModelSets}
              className="w-full text-left text-blue-400 hover:text-blue-500 focus:outline-none flex items-center"
            >
              <span className="mr-2">
                {isModelSetsOpen ? (
                  <svg
                    className="w-4 h-4 transform rotate-90"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                ) : (
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                )}
              </span>
              <span className="font-semibold text-lg">Model Sets</span>
            </button>

            {/* Collapsible Model Sets */}
            {isModelSetsOpen && (
              <div className="mt-4 border-t border-gray-700 pt-4 max-h-80 overflow-y-auto custom-scrollbar">
                {sessionData.model_sets && sessionData.model_sets.length > 0 ? (
                  sessionData.model_sets.map((modelSet, index) => (
                    <div
                      key={index}
                      className="mb-4 p-4 bg-[#2b2b2b] rounded border border-gray-700"
                    >
                      <h4 className="font-semibold text-gray-200 mb-3">
                        Model Set {index + 1}
                      </h4>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs font-mono text-gray-300">
                        <p>X Shape:</p>
                        <p className="text-right">{formatShape(modelSet.X)}</p>
                        <p>y Shape:</p>
                        <p className="text-right">{formatShape(modelSet.y)}</p>
                        <p>X_train Shape:</p>
                        <p className="text-right">{formatShape(modelSet.X_train)}</p>
                        <p>X_test Shape:</p>
                        <p className="text-right">{formatShape(modelSet.X_test)}</p>
                        <p>y_train Shape:</p>
                        <p className="text-right">{formatShape(modelSet.y_train)}</p>
                        <p>y_test Shape:</p>
                        <p className="text-right">{formatShape(modelSet.y_test)}</p>
                        <p>X_train_scaled Shape:</p>
                        <p className="text-right">
                          {formatShape(modelSet.X_train_scaled)}
                        </p>
                        <p>X_test_scaled Shape:</p>
                        <p className="text-right">
                          {formatShape(modelSet.X_test_scaled)}
                        </p>
                        <p>y_train_scaled Shape:</p>
                        <p className="text-right">
                          {formatShape(modelSet.y_train_scaled)}
                        </p>
                        <p>y_test_scaled Shape:</p>
                        <p className="text-right">
                          {formatShape(modelSet.y_test_scaled)}
                        </p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-400">No Model Sets Available</p>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default TrainingSessionCard;