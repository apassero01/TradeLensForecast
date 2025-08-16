import { useState, useCallback } from 'react';

export interface Position {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface WindowState {
  id: string;
  entityId: string;
  position: Position;
  size: Size;
  zIndex: number;
}

interface UseWindowManagerReturn {
  windows: WindowState[];
  openWindow: (entityId: string) => void;
  closeWindow: (windowId: string) => void;
  focusWindow: (windowId: string) => void;
  updateWindowPosition: (windowId: string, position: Position) => void;
  updateWindowSize: (windowId: string, size: Size) => void;
  loadPersistedWindows: (persistedWindows: any[]) => void;
}

// Default window sizes based on entity type
const getDefaultWindowSize = (entityType?: string): Size => {
  switch (entityType) {
    case 'document':
      return { width: 800, height: 600 };
    case 'api_model':
    case 'agent':
      return { width: 700, height: 500 };
    case 'view':
      return { width: 400, height: 600 };
    default:
      return { width: 600, height: 500 };
  }
};

// Calculate smart positioning for new windows
const calculateNewPosition = (existingWindows: WindowState[], windowSize: Size): Position => {
  if (existingWindows.length === 0) {
    // First window: center of viewport
    return {
      x: Math.max(0, (window.innerWidth - windowSize.width) / 2),
      y: Math.max(0, (window.innerHeight - windowSize.height) / 2 - 50), // Account for header
    };
  }

  // Cascade pattern: offset by 30px
  const baseX = 100;
  const baseY = 100;
  const offset = 30;
  const cascadePosition = existingWindows.length * offset;

  let x = baseX + cascadePosition;
  let y = baseY + cascadePosition;

  // Reset cascade if we're getting too close to viewport edge
  if (x + windowSize.width > window.innerWidth - 50 || y + windowSize.height > window.innerHeight - 100) {
    x = baseX;
    y = baseY;
  }

  return { x, y };
};

export const useWindowManager = (): UseWindowManagerReturn => {
  const [windows, setWindows] = useState<WindowState[]>([]);

  const [nextZIndex, setNextZIndex] = useState(2);

  const focusWindow = useCallback((windowId: string) => {
    setWindows(prev => prev.map(w => 
      w.id === windowId 
        ? { ...w, zIndex: nextZIndex }
        : w
    ));
    setNextZIndex(prev => prev + 1);
  }, [nextZIndex]);

  const openWindow = useCallback((entityId: string) => {
    // Don't open duplicate windows for the same entity
    if (windows.some(w => w.entityId === entityId)) {
      // Instead, focus the existing window
      const existingWindow = windows.find(w => w.entityId === entityId);
      if (existingWindow) {
        focusWindow(existingWindow.id);
      }
      return;
    }

    const windowId = `window-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const size = getDefaultWindowSize(); // Will determine based on entity type later
    const position = calculateNewPosition(windows, size);

    const newWindow: WindowState = {
      id: windowId,
      entityId,
      position,
      size,
      zIndex: nextZIndex,
    };

    setWindows(prev => [...prev, newWindow]);
    setNextZIndex(prev => prev + 1);
  }, [windows, nextZIndex, focusWindow]);

  const closeWindow = useCallback((windowId: string) => {
    setWindows(prev => prev.filter(w => w.id !== windowId));
  }, []);

  const updateWindowPosition = useCallback((windowId: string, position: Position) => {
    setWindows(prev => prev.map(w => 
      w.id === windowId 
        ? { ...w, position }
        : w
    ));
  }, []);

  const updateWindowSize = useCallback((windowId: string, size: Size) => {
    setWindows(prev => prev.map(w => 
      w.id === windowId 
        ? { ...w, size }
        : w
    ));
  }, []);

  const loadPersistedWindows = useCallback((persistedWindows: any[]) => {
    const restoredWindows: WindowState[] = persistedWindows.map((persistedWindow, index) => {
      const windowId = `restored-${Date.now()}-${index}`;
      return {
        id: windowId,
        entityId: persistedWindow.entityId,
        position: persistedWindow.position || { x: 100 + index * 30, y: 100 + index * 30 },
        size: persistedWindow.size || getDefaultWindowSize(),
        zIndex: index + 1,
      };
    });

    setWindows(restoredWindows);
    setNextZIndex(restoredWindows.length + 1);
  }, []);

  return {
    windows,
    openWindow,
    closeWindow,
    focusWindow,
    updateWindowPosition,
    updateWindowSize,
    loadPersistedWindows,
  };
};
