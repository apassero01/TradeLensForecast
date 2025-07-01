import React, { useState } from 'react';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface CalendarEventDetailsProps {
  data?: CalendarEventData;
  sendStrategyRequest: (strategyRequest: any) => void;
  updateEntity: (entityId: string, data: any) => void;
  viewEntityId: string;
  parentEntityId: string;
}

interface CalendarEventData {
  title?: string;
  start_time?: string;
  end_time?: string;
  description?: string;
  location?: string;
  date?: string;
}

export default function CalendarEventDetails({
  data,
  sendStrategyRequest,
  parentEntityId,
  viewEntityId,
  updateEntity,
}: CalendarEventDetailsProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    title: data?.title || '',
    date: data?.date || '',
    start_time: data?.start_time || '',
    end_time: data?.end_time || '',
    description: data?.description || '',
    location: data?.location || ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCancel = () => {
    setFormData({
      title: data?.title || '',
      date: data?.date || '',
      start_time: data?.start_time || '',
      end_time: data?.end_time || '',
      description: data?.description || '',
      location: data?.location || ''
    });
    setIsEditing(false);
  };

  const handleSubmit = async(e: React.FormEvent) => {

    e.preventDefault();

    const updated_attributes = {};
    Object.entries(formData).forEach(([key, value]) => {
      updated_attributes[key] = value;
    });

    sendStrategyRequest(StrategyRequests.builder()
          .withStrategyName('SetAttributesStrategy')
          .withTargetEntity(parentEntityId)
          .withParams({
            attribute_map: updated_attributes
          })
          .build());
    setIsEditing(false);
  };

  const formatDateTime = (dateTimeStr: string) => {
    if (!dateTimeStr) return '';
    const date = new Date(dateTimeStr);
    return date.toLocaleString();
  };

  return (
    <div className="flex flex-col h-full w-full bg-gray-800 text-white p-4 overflow-hidden">
      <div className="flex-shrink-0 mb-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white mb-2">
            {isEditing ? 'Edit Event' : `Details for ${data?.title || 'Untitled Event'}`}
          </h1>
          <button
            onClick={() => setIsEditing(!isEditing)}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            {isEditing ? 'Cancel' : 'Edit'}
          </button>
        </div>

        {isEditing ? (
          <form onSubmit={handleSubmit} className="space-y-4 mt-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Event Title
              </label>
              <input
                type="text"
                name="title"
                value={formData.title}
                onChange={handleInputChange}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter event title"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Date
              </label>
              <input
                type="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  Start Time
                </label>
                <input
                  type="time"
                  name="start_time"
                  value={formData.start_time}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">
                  End Time
                </label>
                <input
                  type="time"
                  name="end_time"
                  value={formData.end_time}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Location
              </label>
              <input
                type="text"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter location"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Description
              </label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                rows={3}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter event description"
              />
            </div>

            <div className="flex gap-2">
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
              >
                Save Changes
              </button>
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <div className="space-y-4 mt-4">
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">📅</span>
              <span className="text-gray-300">
                <b>Date:</b>  {data?.date || 'No date'}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">🕐</span>
              <span className="text-gray-300">
                <b>Time:</b> {data?.start_time || 'No start time'} - {data?.end_time || 'No end time'}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-gray-400">📍</span>
              <span className="text-gray-300">
                <b>Location:</b> {data?.location || 'No location'}
              </span>
            </div>
            
            <div className="flex items-start space-x-2">
              <span className="text-gray-400 mt-1">📝</span>
              <span className="text-gray-300">
                <b>Description:</b> {data?.description || 'No description'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 