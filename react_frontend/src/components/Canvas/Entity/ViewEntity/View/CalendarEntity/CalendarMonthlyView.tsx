import React, { useState } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import { eachDayOfInterval, endOfMonth, startOfMonth, startOfWeek } from 'date-fns';
import clsx from 'clsx';

import useRenderStoredView from '../../../../../../hooks/useRenderStoredView';
import { IoClose, IoCheckbox, IoSquareOutline } from 'react-icons/io5';
import { StrategyRequests } from '../../../../../../utils/StrategyRequestBuilder';

interface CalendarMonthlyViewProps {
    data?: CalendarMonthlyViewData;
    sendStrategyRequest: (strategyRequest: any) => void;
    updateEntity: (entityId: string, data: any) => void;
    viewEntityId: string;
    parentEntityId: string;
}

interface CalendarMonthlyViewData {
    currentMonth?: string; // ISO date string for the current month
}

// Add this interface for the date object
interface CalendarDay {
  date: Date;
  events: any[];  // or create a proper Event interface if you have one
  isCurrentMonth: boolean;
}

export default function CalendarMonthlyView({
    data,
    sendStrategyRequest,
    updateEntity,
    viewEntityId,
    parentEntityId,
}: CalendarMonthlyViewProps) {
    // State management
    const [currentDate, setCurrentDate] = useState<Date>(new Date());
    const [currentYear, setCurrentYear] = useState(currentDate.getFullYear());
    const [currentMonth, setCurrentMonth] = useState(currentDate.getMonth()); // 0 = Ja
    const [showEventModal, setShowEventModal] = useState(false);
    const [showDayModal, setShowDayModal] = useState(false);
    const [selectedDay, setSelectedDay] = useState<string>('');
    const [draggedOverDay, setDraggedOverDay] = useState<string | null>(null); // For drag-over visual feedback

    // Get all event children of the calendar
    const eventChildren = useRecoilValue(
        childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.CALENDAR_EVENT })
    ) as any[];

    // Create a mapping of events by date string (YYYY-MM-DD)
    const eventsByDate = React.useMemo(() => {
        const map: { [key: string]: any[] } = {};
        eventChildren.forEach(event => {
            const eventDate = new Date(event.start_time);
            const dateKey = eventDate.toISOString().split('T')[0]; // YYYY-MM-DD
            if (!map[dateKey]) {
                map[dateKey] = [];
            }
            map[dateKey].push(event);
        });
        return map;
    }, [eventChildren]);


    const DAYS_OF_WEEK = [
        { key: 'sunday', label: 'Sunday' },
        { key: 'monday', label: 'Monday' },
        { key: 'tuesday', label: 'Tuesday' },
        { key: 'wednesday', label: 'Wednesday' },
        { key: 'thursday', label: 'Thursday' },
        { key: 'friday', label: 'Friday' },
        { key: 'saturday', label: 'Saturday' },
    ];

    const firstDayOfMonth = startOfMonth(data?.currentMonth? new Date(data.currentMonth) : new Date());
    const lastDayOfMonth = endOfMonth(data?.currentMonth? new Date(data.currentMonth) : new Date());

    // Then modify createCalendarDays to use this map
    const createCalendarDays = (firstDayOfMonth: Date, lastDayOfMonth: Date): CalendarDay[] => {
        const days = eachDayOfInterval({ 
            start: firstDayOfMonth, 
            end: lastDayOfMonth 
        });

        return days.map(day => {
            const dateKey = day.toISOString().split('T')[0];
            return {
                date: day,
                events: eventsByDate[dateKey] || [],
                isCurrentMonth: true
            };
        });
    };

    const renderEventsForDay = (day: CalendarDay) => {
        return day.events.map(event => (
            <div key={event.entity_id} className="text-xs bg-blue-500 rounded p-1 mb-1 truncate" title={event.title}>
                {event.title}
            </div>
        ));
    };

    // Use it in your component
    const calendarDays = createCalendarDays(firstDayOfMonth, lastDayOfMonth);

    const startingDayIndex = startOfWeek(firstDayOfMonth);

    if (!data) {
        return (
        <div className="flex items-center justify-center h-full text-gray-500">
            Loading calendar monthly view...
        </div>
        );
    }

    return (
        <div className="flex flex-col h-full w-full bg-gray-800 text-white p-4 overflow-hidden">
            <div className="flex-shrink-0 mb-6">
                <h1 className="text-2xl font-bold text-white mb-2">
                    Event Calendar
                </h1>
                <div className = "grid grid-cols-7 gap-2">
                    {DAYS_OF_WEEK.map((day) => (
                        <div key={day.key} className="text-center text-sm font-medium text-gray-400">
                            {day.label}
                        </div>
                    ))}
                    {Array.from({ length: startingDayIndex.getDay() }).map((_, index) => (
                        <div key={`empty-${index}`} className="border border-gray-600 text-center text-sm font-medium text-gray-400"></div>
                    ))}
                    {calendarDays.map((day) => (
                        <div 
                            className={clsx(
                                "border border-gray-600 p-2 min-h-[100px] flex flex-col",
                                day.date.getDate() === currentDate.getDate() && "bg-gray-500"
                            )}
                        >
                            <div className="text-sm font-medium text-gray-400 flex-shrink-0">
                                {day.date.getDate()}
                            </div>
                            <div className="mt-1 flex-grow overflow-y-auto">
                                {renderEventsForDay(day)}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
            <div className="flex-1 overflow-y-auto">
            </div>
        </div>
    );
}