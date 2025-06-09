import React, { memo, useState } from 'react';
import EntityNodeBase from '../EntityNodeBase';
import viewComponents from './View/viewComponents';
import { useRecoilValue } from 'recoil';
import { nodeSelectorFamily } from '../../../../state/entitiesSelectors';
import useRenderStoredView from '../../../../hooks/useRenderStoredView';
import { StrategyRequests } from '../../../../utils/StrategyRequestBuilder';
import { FaCog } from 'react-icons/fa'; // Import gear icon from react-icons

function ViewEntity({ data, sendStrategyRequest, updateEntity }) {
  const parentIds = data.parent_ids;
  const parentEntityId = parentIds?.length > 0 ? parentIds[0] : null;
  const parentEntity = useRecoilValue(nodeSelectorFamily(parentEntityId));
  const viewData = useRenderStoredView(data.entityId, sendStrategyRequest, updateEntity);

  const [selectedViewType, setSelectedViewType] = useState('');
  const [mappings, setMappings] = useState({});
  const [isConfiguring, setIsConfiguring] = useState(!viewData); // Show config if no viewData

  if (!parentEntity) {
    return <div className="p-4 text-red-500">No parent entity found.</div>;
  }

  const parentAttributes = Object.keys(parentEntity.data || {});

  const viewComponentNames = {
    histogram: 'Histogram',
    linegraph: 'Line Graph',
    stockchart: 'Stock Chart',
    multiline: 'Multi Line',
    editor: 'Editor',
    chat: 'Chat Screen',
    photo: 'Photo Display',
  };

  const handleSave = () => {
    if (!selectedViewType) {
      alert('Please select a view component');
      return;
    }
    const attributes = {
      view_component_type: selectedViewType,
      parent_attributes: mappings,
    };
    const request = StrategyRequests.setAttributes(data.entityId, attributes, false);
    sendStrategyRequest(request);
    setIsConfiguring(false); // Switch back to view after saving
  };

  const toggleConfigure = () => {
    setIsConfiguring(!isConfiguring);
  };

  const configurationForm = (
    <div className="p-4 bg-gray-800 rounded-lg shadow-md text-gray-300">
      <h2 className="text-lg font-bold mb-4">Configure View</h2>

      {/* View Component Selection and Submit Button */}
      <div className="flex items-center mb-6">
        <div className="flex-grow mr-4">
          <select
            className="w-full p-2 bg-gray-700 text-white border-2 border-gray-600 rounded-md focus:outline-none focus:border-blue-500 transition-all duration-200"
            value={selectedViewType}
            onChange={(e) => setSelectedViewType(e.target.value)}
          >
            <option value="">view component</option>
            {Object.entries(viewComponents).map(([key]) => (
              <option key={key} value={key}>
                {viewComponentNames[key] || key}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={handleSave}
          className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-500 transition-all duration-200"
        >
          Save
        </button>
      </div>

      {/* Attribute Mappings */}
      <div className="mb-6 max-h-64 overflow-y-auto nowheel px-4">
        <label className="block mb-2 text-sm font-medium">Map Parent Attributes</label>
        {parentAttributes.length > 0 ? (
          parentAttributes.map((attr) => (
            <div key={attr} className="flex items-center mb-2">
              <input
                type="checkbox"
                checked={attr in mappings}
                onChange={(e) => {
                  if (e.target.checked) {
                    setMappings({ ...mappings, [attr]: attr });
                  } else {
                    const { [attr]: _, ...rest } = mappings;
                    setMappings(rest);
                  }
                }}
                className="mr-2 h-4 w-4 text-blue-600 border-gray-600 rounded"
              />
              <span className="mr-4 w-32">{attr}</span>
              {attr in mappings && (
                <input
                  type="text"
                  value={mappings[attr]}
                  onChange={(e) => setMappings({ ...mappings, [attr]: e.target.value })}
                  placeholder="View prop name"
                  className="flex-grow p-2 bg-gray-700 text-white border-2 border-gray-600 rounded-md focus:outline-none focus:border-blue-500 transition-all duration-200"
                />
              )}
            </div>
          ))
        ) : (
          <p className="text-gray-400">No parent attributes available.</p>
        )}
      </div>
    </div>
  );

  return (
    <EntityNodeBase data={data} updateEntity={updateEntity}>
      {() => (
        <div className="relative flex-grow h-full w-full my-4 -mx-6 overflow-hidden">
          {/* Gear Icon for Toggling */}
          {viewData && (
            <button
              onClick={toggleConfigure}
              className="absolute top-2 left-2 z-10 p-2 bg-gray-700 text-white rounded-full hover:bg-gray-600 transition-all duration-200"
            >
              <FaCog className="text-xl" />
            </button>
          )}
          <div className="h-full w-full px-6">
            {isConfiguring ? configurationForm : viewData}
          </div>
        </div>
      )}
    </EntityNodeBase>
  );
}

export default memo(ViewEntity);