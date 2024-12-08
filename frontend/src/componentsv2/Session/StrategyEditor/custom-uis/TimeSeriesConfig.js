import React from 'react';

const TimeSeriesConfig = ({ config, onChange }) => {
  const handleChange = (field, value) => {
    onChange({
      ...config,
      [field]: value
    });
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Time Column
        </label>
        <input
          type="text"
          value={config.timeColumn || ''}
          onChange={(e) => handleChange('timeColumn', e.target.value)}
          className="w-full bg-gray-700 text-white px-3 py-2 rounded"
        />
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Aggregation Period
        </label>
        <select
          value={config.aggregation || ''}
          onChange={(e) => handleChange('aggregation', e.target.value)}
          className="w-full bg-gray-700 text-white px-3 py-2 rounded"
        >
          <option value="1d">Daily</option>
          <option value="1h">Hourly</option>
          <option value="1w">Weekly</option>
          <option value="1m">Monthly</option>
        </select>
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-1">
          Features to Aggregate
        </label>
        <div className="space-y-2">
          {['mean', 'sum', 'min', 'max'].map(agg => (
            <label key={agg} className="flex items-center">
              <input
                type="checkbox"
                checked={config.aggregations?.includes(agg) || false}
                onChange={(e) => {
                  const current = config.aggregations || [];
                  const updated = e.target.checked
                    ? [...current, agg]
                    : current.filter(x => x !== agg);
                  handleChange('aggregations', updated);
                }}
                className="mr-2"
              />
              <span className="text-white capitalize">{agg}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TimeSeriesConfig; 