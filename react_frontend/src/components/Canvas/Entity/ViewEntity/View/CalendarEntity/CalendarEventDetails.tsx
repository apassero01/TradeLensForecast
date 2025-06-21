import React, { useState } from 'react';
import { IoCalendar, IoTime, IoLocation, IoDocumentText } from 'react-icons/io5';
import type { IconType } from 'react-icons';

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
  console.log(data);
  const [isEditing, setIsEditing] = useState(false);
  const [editedData, setEditedData] = useState(data || {});


  const handleCancel = () => {
    setEditedData(data || {});
    setIsEditing(false);
  };

  const formatDateTime = (dateTimeStr: string) => {
    if (!dateTimeStr) return '';
    const date = new Date(dateTimeStr);
    return date.toLocaleString();
  };

  return (<div>
    <h1>{data?.title || 'Untitled Event'}</h1>
    <p>{data?.date || 'No date'}</p>
    <p>{data?.start_time || 'No start time'}</p>
    <p>{data?.end_time || 'No end time'}</p>
    <p>{data?.description || 'No description'}</p>
    <p>{data?.location || 'No location'}</p>
    </div>);
} 