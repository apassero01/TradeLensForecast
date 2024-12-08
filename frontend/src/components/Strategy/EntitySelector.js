import React, { useState, useEffect } from 'react';
import { fetchAvailableEntities } from '../api';

const EntitySelector = ({ value, onChange }) => {
  const [entities, setEntities] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadEntities = async () => {
      try {
        const availableEntities = await fetchAvailableEntities();
        setEntities(availableEntities);
        setLoading(false);
      } catch (err) {
        setError('Failed to load entities');
        setLoading(false);
      }
    };

    loadEntities();
  }, []);

  if (loading) return <div className="text-gray-400">Loading entities...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-300">
        Select Entity Type
      </label>
      <select
        value={value}
        onChange={(e) => {
          const selectedEntity = entities[e.target.value];
          onChange({
            entity_name: selectedEntity.name,
            entity_class: selectedEntity.class_path
          });
        }}
        className="w-full bg-gray-700/50 border border-gray-600 rounded-md px-3 py-2 text-sm text-gray-300
                 focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Select an entity</option>
        {Object.entries(entities).map(([key, entity]) => (
          <option key={key} value={key}>
            {entity.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default EntitySelector; 