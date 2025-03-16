import { useReactFlow } from '@xyflow/react';
import { useState, useEffect, useCallback } from 'react';


export function useAdjustedPointerPosition() {
    const { transform } = useReactFlow();
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const zoom = transform ? transform[2] : 1;
    
    const handlePointerMove = useCallback((e) => {
      // Convert the screen coordinates to node coordinates by dividing by zoom.
      setPosition({
        x: e.clientX / zoom,
        y: e.clientY / zoom,
      });
    }, [zoom]);
    
    useEffect(() => {
      window.addEventListener('pointermove', handlePointerMove);
      return () => {
        window.removeEventListener('pointermove', handlePointerMove);
      };
    }, [handlePointerMove]);
    
    return position;
  }