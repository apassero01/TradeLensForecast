import React, { useEffect } from 'react';
import { useRecoilState } from 'recoil';
import { notificationAtom } from '../../state/notificationAtom';

export function Notification() {
    const [notification, setNotification] = useRecoilState(notificationAtom);

    useEffect(() => {
        if (notification) {
            const timer = setTimeout(() => {
                setNotification(null);
            }, 5000);
            return () => clearTimeout(timer);
        }
    }, [notification, setNotification]);

    if (!notification) return null;

    const bgColor = {
        error: 'bg-red-500',
        success: 'bg-green-500',
        info: 'bg-blue-500'
    }[notification.type];

    // Format the message if it's an object or array
    const formatMessage = (message) => {
        if (typeof message === 'string') return message;
        if (Array.isArray(message)) return message.join('\n');
        if (typeof message === 'object') return JSON.stringify(message, null, 2);
        return String(message);
    };

    return (
        <div className={`fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg max-w-md whitespace-pre-wrap`}>
            {formatMessage(notification.message)}
        </div>
    );
} 