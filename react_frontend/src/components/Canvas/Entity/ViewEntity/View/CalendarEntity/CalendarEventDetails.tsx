import React, { useState, useRef } from 'react';
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
  const [formData, setFormData] = useState({
    title: data?.title || '',
    date: data?.date || '',
    start_time: data?.start_time || '',
    end_time: data?.end_time || '',
    description: data?.description || '',
    location: data?.location || ''
  });

  const [editingField, setEditingField] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFieldClick = (field: string) => {
    setEditingField(field);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  const handleFieldBlur = (field: string) => {
    setEditingField(null);
    handleSubmitField(field);
  };

  const handleFieldKeyDown = (e: React.KeyboardEvent, field: string) => {
    if (e.key === 'Enter' && field !== 'description') {
      setEditingField(null);
      handleSubmitField(field);
    } else if (e.key === 'Escape') {
      setFormData(prev => ({ ...prev, [field]: data?.[field as keyof CalendarEventData] || '' }));
      setEditingField(null);
    }
  };

  const handleSubmitField = (field: string) => {
    // Only update the single field
    const updated_attributes = { [field]: formData[field as keyof typeof formData] };
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('SetAttributesStrategy')
      .withTargetEntity(parentEntityId)
      .withParams({ attribute_map: updated_attributes })
      .build());
  };

  const handleDeleteEvent = () => {
    sendStrategyRequest(StrategyRequests.builder()
      .withStrategyName('RemoveEntityStrategy')
      .withTargetEntity(parentEntityId)
      .build());
  };

  const formatDateTime = (dateTimeStr: string) => {
    if (!dateTimeStr) return '';
    const date = new Date(dateTimeStr);
    return date.toLocaleString();
  };

  return (
    <div className="flex flex-col h-full w-full bg-gray-800 text-white p-4 overflow-hidden">
      <div className="flex-shrink-0 mb-6">
        <h1 className="text-2xl font-bold text-white mb-2">
          {formData.title ? `Details for ${formData.title}` : 'Event Details'}
        </h1>
        <div className="space-y-4 mt-4">
          {/* Date */}
          <div className="flex items-center space-x-2">
            <span className="text-gray-400">📅</span>
            {editingField === 'date' ? (
              <input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                type="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                onBlur={() => handleFieldBlur('date')}
                onKeyDown={e => handleFieldKeyDown(e, 'date')}
                className="w-40 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
              />
            ) : (
              <span
                className="text-gray-300 cursor-pointer hover:underline"
                onClick={() => handleFieldClick('date')}
              >
                <b>Date:</b>  {formData.date || 'No date'}
              </span>
            )}
          </div>

          {/* Time */}
          <div className="flex items-center space-x-2">
            <span className="text-gray-400">🕐</span>
            <span className="text-gray-300">
              <b>Time:</b> {' '}
              {editingField === 'start_time' ? (
                <input
                  ref={inputRef as React.RefObject<HTMLInputElement>}
                  type="time"
                  name="start_time"
                  value={formData.start_time}
                  onChange={handleInputChange}
                  onBlur={() => handleFieldBlur('start_time')}
                  onKeyDown={e => handleFieldKeyDown(e, 'start_time')}
                  className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white mr-2"
                />
              ) : (
                <span
                  className="cursor-pointer hover:underline"
                  onClick={() => handleFieldClick('start_time')}
                >
                  {formData.start_time || 'No start time'}
                </span>
              )}
              {' - '}
              {editingField === 'end_time' ? (
                <input
                  ref={inputRef as React.RefObject<HTMLInputElement>}
                  type="time"
                  name="end_time"
                  value={formData.end_time}
                  onChange={handleInputChange}
                  onBlur={() => handleFieldBlur('end_time')}
                  onKeyDown={e => handleFieldKeyDown(e, 'end_time')}
                  className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                />
              ) : (
                <span
                  className="cursor-pointer hover:underline"
                  onClick={() => handleFieldClick('end_time')}
                >
                  {formData.end_time || 'No end time'}
                </span>
              )}
            </span>
          </div>

          {/* Location */}
          <div className="flex items-center space-x-2">
            <span className="text-gray-400">📍</span>
            {editingField === 'location' ? (
              <input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                type="text"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                onBlur={() => handleFieldBlur('location')}
                onKeyDown={e => handleFieldKeyDown(e, 'location')}
                className="w-64 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter location"
              />
            ) : (
              <span
                className="text-gray-300 cursor-pointer hover:underline"
                onClick={() => handleFieldClick('location')}
              >
                <b>Location:</b> {formData.location || 'No location'}
              </span>
            )}
          </div>

          {/* Description */}
          <div className="flex items-start space-x-2">
            <span className="text-gray-400 mt-1">📝</span>
            {editingField === 'description' ? (
              <textarea
                ref={inputRef as React.RefObject<HTMLTextAreaElement>}
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                onBlur={() => handleFieldBlur('description')}
                onKeyDown={e => handleFieldKeyDown(e, 'description')}
                rows={3}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter event description"
              />
            ) : (
              <span
                className="text-gray-300 cursor-pointer hover:underline"
                onClick={() => handleFieldClick('description')}
              >
                <b>Description:</b> {formData.description || 'No description'}
              </span>
            )}
          </div>

          {/* Title (at the top, but also allow inline edit) */}
          <div className="flex items-center space-x-2 mt-4">
            <span className="text-gray-400">🏷️</span>
            {editingField === 'title' ? (
              <input
                ref={inputRef as React.RefObject<HTMLInputElement>}
                type="text"
                name="title"
                value={formData.title}
                onChange={handleInputChange}
                onBlur={() => handleFieldBlur('title')}
                onKeyDown={e => handleFieldKeyDown(e, 'title')}
                className="w-64 px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white"
                placeholder="Enter event title"
              />
            ) : (
              <span
                className="text-gray-300 cursor-pointer hover:underline"
                onClick={() => handleFieldClick('title')}
              >
                <b>Title:</b> {formData.title || 'Untitled Event'}
              </span>
            )}
          </div>

          
        </div>
      </div>

      {/* Delete Button Section */}
      <div className="flex-shrink-0 mt-auto pt-6 border-t border-gray-700">
        {!showDeleteConfirm ? (
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="w-full px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-md transition-colors duration-200 flex items-center justify-center space-x-2"
          >
            <span>Delete Event</span>
          </button>
        ) : (
          <div className="space-y-3">
            <div className="text-center text-gray-300 text-sm">
              Are you sure you want to delete "{formData.title || 'this event'}"?
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 w-40 px-4 py-3 bg-gray-800 border border-gray-600 hover:bg-gray-700 text-white font-medium rounded-md transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteEvent}
                className="flex-1 px-4 py-3 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-md transition-colors duration-200"
              >
                Delete
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 