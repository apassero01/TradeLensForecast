import { useState, useEffect } from 'react';
import { entityApi } from '../api/entityApi';


export function useStrategyRegistry() {
    const [registry, setRegistry] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function fetchRegistry() {
          try {
            const result = await entityApi.getStrategyRegistry(); 
            setRegistry(result);
            setLoading(false);
          } catch (err) {
            console.error('Failed to fetch strategy registry', err);
            setError(err.message);
            setLoading(false);
          }
        }
        fetchRegistry();
      }, []);

    return { registry, loading, error };
}