import React, { useState, useEffect } from 'react';
import { fetchSequenceSetsMetaData, fetchAllXFeatures, fetchAllyFeatures } from './SequenceSetSelectorApi';
import SelectionBox from './SelectionBox';
import DateInput from './DateInput';

const SequenceSetSelector = ({ value, onChange }) => {
  const [allModelSets, setAllModelSets] = useState([]);
  const [allXFeatures, setAllXFeatures] = useState([]);
  const [allYFeatures, setAllYFeatures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [sequenceSets, xFeatures, yFeatures] = await Promise.all([
          fetchSequenceSetsMetaData(),
          fetchAllXFeatures(),
          fetchAllyFeatures()
        ]);
        
        setAllModelSets(sequenceSets);
        setAllXFeatures(xFeatures);
        setAllYFeatures(yFeatures);
        setLoading(false);
      } catch (err) {
        setError('Failed to load configuration data');
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleChange = (field, newValue) => {
    const updatedConfig = {
      ...value,
      X_features: value?.X_features || null,
      y_features: value?.y_features || null,
      model_set_configs: value?.model_set_configs || null,
      dataset_type: "stock"
    };

    switch (field) {
      case 'modelSets':
        updatedConfig.model_set_configs = newValue.map(set => ({
          id: set.id,
          ticker: set.ticker,
          sequence_length: set.sequence_length,
          interval: set.interval,
          start_timestamp: updatedConfig.model_set_configs?.[0]?.start_timestamp || ''
        }));
        break;
      case 'xFeatures':
        updatedConfig.X_features = newValue.map(feature => feature.name);
        break;
      case 'yFeatures':
        updatedConfig.y_features = newValue.map(feature => feature.name);
        break;
      case 'startDate':
        if (updatedConfig.model_set_configs) {
          updatedConfig.model_set_configs = updatedConfig.model_set_configs.map(config => ({
            ...config,
            start_timestamp: newValue
          }));
        }
        break;
    }

    onChange(updatedConfig);
  };

  if (loading) return <div className="text-gray-400">Loading configuration...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="space-y-4">
      <DateInput
        label="Select Start Date"
        selectedDate={value?.model_set_configs?.[0]?.start_timestamp || ''}
        onDateChange={(date) => handleChange('startDate', date)}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <SelectionBox
            label="SequenceSets"
            items={allModelSets}
            itemKey="id"
            displayText={(item) => `${item.ticker}-${item.sequence_length}-${item.interval}`}
            selectedItems={value?.model_set_configs?.map(config => 
              allModelSets.find(set => set.id === config.id)
            ).filter(Boolean) || []}
            onSelectionChange={(set, isSelected) => {
              const currentSets = value?.model_set_configs?.map(config => 
                allModelSets.find(set => set.id === config.id)
              ).filter(Boolean) || [];
              
              const newSets = isSelected
                ? [...currentSets, set]
                : currentSets.filter(s => s.id !== set.id);
              
              handleChange('modelSets', newSets);
            }}
          />
        </div>

        <div>
          <SelectionBox
            label="Select X Features"
            items={allXFeatures}
            itemKey="name"
            displayText={(item) => item.name}
            selectedItems={value?.X_features?.map(name => ({ name })) || []}
            onSelectionChange={(feature, isSelected) => {
              const currentFeatures = value?.X_features?.map(name => ({ name })) || [];
              const newFeatures = isSelected
                ? [...currentFeatures, feature]
                : currentFeatures.filter(f => f.name !== feature.name);
              handleChange('xFeatures', newFeatures);
            }}
          />
        </div>

        <div>
          <SelectionBox
            label="Select Y Features"
            items={allYFeatures}
            itemKey="name"
            displayText={(item) => item.name}
            selectedItems={value?.y_features?.map(name => ({ name })) || []}
            onSelectionChange={(feature, isSelected) => {
              const currentFeatures = value?.y_features?.map(name => ({ name })) || [];
              const newFeatures = isSelected
                ? [...currentFeatures, feature]
                : currentFeatures.filter(f => f.name !== feature.name);
              handleChange('yFeatures', newFeatures);
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default SequenceSetSelector; 