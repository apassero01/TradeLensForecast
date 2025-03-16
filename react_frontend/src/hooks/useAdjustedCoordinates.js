import { useReactFlow } from '@xyflow/react';
import { useState, useEffect, useCallback } from 'react';

// useAdjustedCoordinates takes a ref to an element (for example, the container of a child)
// and returns its adjusted coordinates relative to the current zoom level.
export function useAdjustedCoordinates(ref) {
  const { transform } = useReactFlow();
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  
  useEffect(() => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      const zoom = transform ? transform[2] : 1;
      // Adjust the element's screen coordinates by the current zoom factor.
      setCoords({
        x: rect.left / zoom,
        y: rect.top / zoom,
      });
    }
  }, [ref, transform]);
  
  return coords;
}