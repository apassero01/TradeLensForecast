import React, { useEffect, useState } from 'react';
import { fetchSequenceSetsMetaData, fetchAllXFeatures, fetchAllyFeatures } from './SequenceSetSelectorApi';
import SelectionBox from './SelectionBox';
import DateInput from './DateInput';

const SequenceSetSelector = ({ value, onChange }) => {
  const [allModelSets, setAllModelSets] = useState([]);
  const [allXFeatures, setAllXFeatures] = useState([]);
  const [allYFeatures, setAllYFeatures] = useState([]);
  const [selectedModelSets, setSelectedModelSets] = useState([]);
  const [selectedXFeatures, setSelectedXFeatures] = useState([]);
  const [selectedYFeatures, setSelectedYFeatures] = useState([]);
  const [startTimestamp, setStartTimestamp] = useState('');
  const [error, setError] = useState(null);

  // Fetch data on component mount
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
      } catch (err) {
        setError('Failed to load configuration data');
      }
    };

    loadData();
  }, []);

  const validateAndUpdate = () => {
    // Clear any previous errors
    setError(null);

    if (!startTimestamp) {
      setError('Please select a start date');
      return false;
    }
    if (selectedModelSets.length === 0) {
      setError('Please select at least one sequence set');
      return false;
    }
    if (selectedXFeatures.length === 0) {
      setError('Please select at least one X feature');
      return false;
    }
    if (selectedYFeatures.length === 0) {
      setError('Please select at least one y feature');
      return false;
    }

    // Only send the selected model sets and their features
    const modelSetConfigs = selectedModelSets.map(set => {
      const { id, ticker, sequence_length, interval } = set;
      return {
        id,
        ticker,
        sequence_length,
        interval,
        start_timestamp: startTimestamp
      };
    });

    console.log('Selected X Features:', selectedXFeatures); // Debug log
    console.log('Selected Y Features:', selectedYFeatures); // Debug log

    // Format the data with only selected features
    const formattedData = {
      X_features: selectedXFeatures.map(feature => feature.name),
      y_features: selectedYFeatures.map(feature => feature.name),
      model_set_configs: modelSetConfigs,
      dataset_type: "stock"
    };

    console.log('Formatted Data:', formattedData); // Debug log
    onChange(formattedData);
    return true;
  };

  // Update parent whenever selections change
  useEffect(() => {
    if (startTimestamp || selectedModelSets.length > 0 || 
        selectedXFeatures.length > 0 || selectedYFeatures.length > 0) {
      validateAndUpdate();
    }
  }, [selectedModelSets, selectedXFeatures, selectedYFeatures, startTimestamp]);

  return (
    <div className="space-y-4">
      {/* Error Display */}
      {error && (
        <div className="text-red-500 text-sm mb-4">
          {error}
        </div>
      )}

      {/* Date Input */}
      <DateInput
        label="Select Start Date"
        onDateChange={setStartTimestamp}
        selectedDate={startTimestamp}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Sequence Set Selection */}
        <div>
          <SelectionBox
            label="SequenceSets"
            items={allModelSets}
            itemKey="id"
            displayText={(item) => `${item.ticker}-${item.sequence_length}-${item.interval}`}
            onSelectionChange={(set, isSelected) => {
              setSelectedModelSets(prev => 
                isSelected 
                  ? [...prev, set]
                  : prev.filter(s => s.id !== set.id)
              );
            }}
            selectedItems={selectedModelSets}
          />
        </div>

        {/* X Features Selection */}
        <div>
          <SelectionBox
            label="Select X Features"
            items={allXFeatures}
            itemKey="name"
            displayText={(item) => item.name}
            onSelectionChange={(feature, isSelected) => {
              setSelectedXFeatures(prev =>
                isSelected
                  ? [...prev, feature]
                  : prev.filter(f => f.name !== feature.name)
              );
            }}
            selectedItems={selectedXFeatures}
          />
        </div>

        {/* Y Features Selection */}
        <div>
          <SelectionBox
            label="Select Y Features"
            items={allYFeatures}
            itemKey="name"
            displayText={(item) => item.name}
            onSelectionChange={(feature, isSelected) => {
              setSelectedYFeatures(prev =>
                isSelected
                  ? [...prev, feature]
                  : prev.filter(f => f.name !== feature.name)
              );
            }}
            selectedItems={selectedYFeatures}
          />
        </div>
      </div>
    </div>
  );
};

export default SequenceSetSelector; 