import React, { useState, useRef, useEffect } from 'react';
import { useRecoilValue } from 'recoil';
import { childrenByTypeSelector } from '../../../../../../state/entitiesSelectors';
import { EntityTypes } from '../../../../Entity/EntityEnum';
import { eachDayOfInterval, endOfMonth, startOfMonth, startOfWeek } from 'date-fns';
import clsx from 'clsx';
import useEntityView from '../../../../../../hooks/useEntityView';

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
    const [showDayModal, setShowDayModal] = useState(false);
    const [selectedDay, setSelectedDay] = useState<CalendarDay | null>(null);
    const [showEventSidebar, setShowEventSidebar] = useState(false);
    const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
    const selectedEventDetailView = useEntityView(selectedEventId, sendStrategyRequest, updateEntity, {}, 'calendar_event_details');
    const [popupPosition, setPopupPosition] = useState<{ top: number; left: number } | null>(null);
    const calendarContainerRef = useRef<HTMLDivElement>(null);
    const popupRef = useRef<HTMLDivElement>(null);

    // Get all event children of the calendar
    const eventChildren = useRecoilValue(
        childrenByTypeSelector({ parentId: parentEntityId, type: EntityTypes.CALENDAR_EVENT })
    ) as any[];

    // Create a mapping of events by date string (YYYY-MM-DD)
    const eventsByDate = React.useMemo(() => {
        const map: { [key: string]: any[] } = {};
        eventChildren.forEach(eventNode => {
            // eventNode is a node object, so we need to access eventNode.data
            const eventData = eventNode.data;
            // Use the date field from CalendarEventEntity
            const dateKey = eventData?.date;
            if (dateKey && !map[dateKey]) {
                map[dateKey] = [];
            }
            if (dateKey) {
                map[dateKey].push(eventData); // Push the actual entity data, not the node
            }
        });
        
        // Sort events within each date by startTime
        Object.keys(map).forEach(dateKey => {
            map[dateKey].sort((a, b) => {
                return a.startTime?.localeCompare(b.startTime) || 0;
            });
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

    const firstDayOfMonth = startOfMonth(new Date(currentYear, currentMonth));
    const lastDayOfMonth = endOfMonth(new Date(currentYear, currentMonth));

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
            <div
                key={event.entity_id}
                className="text-xs bg-blue-500 rounded p-1 mb-1 truncate cursor-pointer"
                onClick={e => {
                    setSelectedEventId(event.entity_id);
                    setShowEventSidebar(true);
                    const dayCell = (e.target as HTMLElement).closest('.calendar-day-cell') as HTMLElement;
                    const container = calendarContainerRef.current;
                    if (dayCell && container) {
                        // Use offsetTop/offsetLeft for zoom-independent positioning
                        const offsetTop = dayCell.offsetTop;
                        const offsetLeft = dayCell.offsetLeft + dayCell.offsetWidth; // to the right of the cell
                        setPopupPosition({
                            top: offsetTop,
                            left: offsetLeft,
                        });
                    }
                }}
                title={event.title}
            >
                {event.title || 'Untitled Event'}
            </div>
        ));
    };

    const changeMonth = (direction: 'previous' | 'next') => {
        if (direction === 'previous') {
            if (currentMonth === 0) {
                // January -> December of previous year
                setCurrentMonth(11);
                setCurrentYear(currentYear - 1);
            } else {
                setCurrentMonth(currentMonth - 1);
            }
        } else {
            if (currentMonth === 11) {
                // December -> January of next year
                setCurrentMonth(0);
                setCurrentYear(currentYear + 1);
            } else {
                setCurrentMonth(currentMonth + 1);
            }
        }
    };

    // Use it in your component
    const calendarDays = createCalendarDays(firstDayOfMonth, lastDayOfMonth);

    // Calculate the starting day offset (0 = Sunday, 1 = Monday, etc.)
    const startingDayOffset = firstDayOfMonth.getDay();
    const endingDayOffset = lastDayOfMonth.getDay();

    // Close popup when clicking outside
    useEffect(() => {
        if (!popupPosition || !selectedEventId) return;

        function handleClickOutside(event: MouseEvent) {
            if (popupRef.current && !popupRef.current.contains(event.target as Node)) {
                setSelectedEventId(null);
                setPopupPosition(null);
            }
        }
        document.addEventListener('pointerdown', handleClickOutside);
        return () => {
            document.removeEventListener('pointerdown', handleClickOutside);
        };
    }, [popupPosition, selectedEventId]);

    if (!data) {
        return (
        <div className="flex items-center justify-center h-full text-gray-500">
            Loading calendar monthly view...
        </div>
        );
    }

    return (
        <div
            ref={calendarContainerRef}
            className="flex flex-col h-full w-full bg-gray-800 text-white p-4 overflow-hidden relative"
        >
            <div className="flex-shrink-0 mb-6 space-y-4">
                <div className="flex items-center justify-between">
                    <button onClick={() => changeMonth('previous')} className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700">Previous</button>
                    <h1 className="text-2xl font-bold text-white mb-2">
                        Your Events for {new Intl.DateTimeFormat('en-US', { month: 'long' }).format(new Date(currentYear, currentMonth))} {currentYear}
                    </h1>
                    <button onClick={() => changeMonth('next')} className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700">Next</button>
                </div>
                <div className = "grid grid-cols-7 gap-2">
                    {DAYS_OF_WEEK.map((day) => (
                        <div key={day.key} className="text-center text-sm font-medium text-gray-400">
                            {day.label}
                        </div>
                    ))}

                    {Array.from({ length: startingDayOffset }).map((_, index) => (
                        <div key={`empty-${index}`} className="border border-gray-600 text-center text-sm font-medium text-gray-400 bg-gray-700"></div>
                    ))}

                    {calendarDays.map((day) => (
                        <div 
                            key={day.date.toISOString()}
                            className={clsx(
                                "calendar-day-cell",
                                "border border-gray-600 p-2 min-h-[100px] flex flex-col cursor-pointer transition-colors duration-200",
                                day.date.getDate() === currentDate.getDate() && 
                                day.date.getMonth() === currentDate.getMonth() && 
                                day.date.getFullYear() === currentDate.getFullYear() && 
                                "bg-gray-500",
                                "hover:bg-gray-600 hover:border-gray-400"
                            )}
                            onDoubleClick={() => {
                                setSelectedDay(day);
                                setShowDayModal(true);
                            }}
                        >
                            <div className="text-sm font-medium text-gray-400 flex-shrink-0">
                                {day.date.getDate()}
                            </div>
                            <div className="mt-1 flex-grow overflow-y-auto">
                                {renderEventsForDay(day)}
                            </div>
                        </div>
                    ))}

                    {Array.from({ length: 6 - endingDayOffset }).map((_, index) => (
                        <div key={`empty-${index}`} className="border border-gray-600 text-center text-sm font-medium text-gray-400 bg-gray-700"></div>
                    ))}


                </div>
            </div>
            
            {/* Day Events Modal */}
            {showDayModal && selectedDay && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 max-h-[80vh] overflow-y-auto">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-xl font-bold text-white">
                                {selectedDay.date.toLocaleDateString('en-US', { 
                                    weekday: 'long', 
                                    year: 'numeric', 
                                    month: 'long', 
                                    day: 'numeric' 
                                })}
                            </h2>
                            <button 
                                onClick={() => setShowDayModal(false)}
                                className="text-gray-400 hover:text-white text-2xl"
                            >
                                ×
                            </button>
                        </div>
                        
                        <div className="space-y-3">
                            {selectedDay.events.length === 0 ? (
                                <p className="text-gray-400 text-center py-4">No events scheduled for this day</p>
                            ) : (
                                selectedDay.events.map((event, index) => (
                                    <div key={event.entity_id || index} 
                                    className={clsx(
                                        "bg-gray-700 rounded-lg p-4",
                                        "hover:bg-gray-600 hover:border-gray-400 cursor-pointer"
                                    )}
                                    onClick={() => {
                                        setSelectedEventId(event.entity_id);
                                        setShowEventSidebar(true);
                                    }}
                                    >
                                        <div className="flex justify-between items-start mb-2">
                                            <h3 className="font-semibold text-white">
                                                {event.title || 'Untitled Event'}
                                            </h3>
                                            {event.startTime && (
                                                <span className="text-sm text-gray-300 bg-gray-600 px-2 py-1 rounded">
                                                    {event.startTime}
                                                </span>
                                            )}
                                        </div>
                                        {event.description && (
                                            <p className="text-gray-300 text-sm">
                                                {event.description}
                                            </p>
                                        )}
                                        {event.location && (
                                            <p className="text-gray-400 text-sm mt-1">
                                                📍 {event.location}
                                            </p>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                        
                        <div className="mt-6 flex justify-end">
                            <button 
                                onClick={() => setShowDayModal(false)}
                                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {popupPosition && selectedEventId && (
                <div
                    ref={popupRef}
                    style={{
                        position: 'absolute',
                        top: popupPosition.top,
                        left: popupPosition.left + 10,
                    }}
                >
                    {selectedEventDetailView}
                </div>
            )}
        </div>
    );
}