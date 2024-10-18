import React, {useEffect} from 'react';
import {fetchSequenceSetsMetaData, fetchAllXFeatures, fetchAllyFeatures} from './api';  // Import the API function
import SelectionBox from './inputs/SelectionBox';  // For displaying the sequence sets
import axios from 'axios';  // Import Axios for making API calls
import qs from 'qs';
import DateInput from "./inputs/DateInput";  // Import qs for serializing data

function TrainingConfiguration({
                                   sessionState,
                                   updateSessionState,
                                   setError,
                                   setLoading,
                               }) {

    // Fetch sequence set metadata when the component mounts
    // Fetch sequence sets data on initial mount, only if they aren't already in the sessionState
    useEffect(() => {
        const loadData = async () => {
            if (sessionState.allSequenceSets.length === 0) {  // Only fetch if the sequence sets aren't already loaded
                const sequenceSets = await fetchSequenceSetsMetaData();
                updateSessionState('allSequenceSets', sequenceSets);
            }
            if (sessionState.allXFeatures.length === 0) {  // Fetch X features if not already loaded
                const xFeatures = await fetchAllXFeatures();
                updateSessionState('allXFeatures', xFeatures);
            }
            if (sessionState.allYFeatures.length === 0) {  // Fetch Y features if not already loaded
                const yFeatures = await fetchAllyFeatures();
                updateSessionState('allYFeatures', yFeatures);
            }
        };

        loadData();
    }, [
        sessionState.allSequenceSets.length,
        sessionState.allXFeatures.length,
        sessionState.allYFeatures.length,
        updateSessionState, // Add this to dependency array or wrap it with useCallback in the parent
        setError,
    ]);

    const handleSubmit = async () => {
        if (sessionState.selectedSets.length === 0) {
            setError('Please select at least one sequence set');
            return;
        }
        if (sessionState.selectedXFeatures.length === 0) {
            setError('Please select at least one X feature');
            return;
        }
        if (sessionState.selectedYFeatures.length === 0) {
            setError('Please select at least one y feature');
            return;
        }

        setLoading(true);
        setError('');

        try {
            if (sessionState.start_timestamp === '') {
                setError('Please select a date');
                setLoading(false);
                return;
            }
            if (sessionState.selectedXFeatures.length === 0) {
                setError('Please select at least one X feature');
                setLoading(false);
                return;
            }
            if (sessionState.selectedYFeatures.length === 0) {
                setError('Please select at least one y feature');
                setLoading(false);
                return;
            }
            // Make the API call to start a new session
            const response = await axios.post(
                'http://localhost:8000/training_manager/start_training_session/',
                qs.stringify({
                    sequence_params: JSON.stringify(sessionState.selectedSets),  // Send selected sets
                    start_timestamp: sessionState.start_timestamp,  // Send selected date
                    X_features: JSON.stringify(sessionState.selectedXFeatures),
                    y_features: JSON.stringify(sessionState.selectedYFeatures)
                }),
                {headers: {'Content-Type': 'application/x-www-form-urlencoded'}}
            );

            updateSessionState('sessionData', response.data);  // Update session data in parent component
            setLoading(false);  // Stop loading
        } catch (err) {
            setError('Failed to start the training session');
            setLoading(false);  // Stop loading
        }
    };

    return (
        <div>
            <h2 className="text-xl font-semibold mb-4">Training Configuration</h2>

            {/* Date Picker */}
            <DateInput
                label="Select Date:"
                onDateChange={(date) => updateSessionState('start_timestamp', date)}
                selectedDate={sessionState.start_timestamp}
            />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">

                {/* Sequence Set Selection */}
                <div>
                    <SelectionBox
                        label="Select Sequence Sets"
                        items={sessionState.allSequenceSets}
                        itemKey="id"
                        displayText={(item) => `${item.ticker} - ${item.sequence_length} - ${item.interval}`}
                        onSelectionChange={(set, isSelected) => {
                            const updatedSets = isSelected
                                ? [...sessionState.selectedSets, set]
                                : sessionState.selectedSets.filter((s) => s.id !== set.id);
                            updateSessionState('selectedSets', updatedSets);
                        }}
                        selectedItems={sessionState.selectedSets}
                    />
                </div>

                {/* X Features Selection */}
                <div>
                    <SelectionBox
                        label="Select X Features"
                        items={sessionState.allXFeatures}
                        itemKey="id"
                        displayText={(item) => item.name}
                        onSelectionChange={(feature, isSelected) => {
                            const updatedFeatures = isSelected
                                ? [...sessionState.selectedXFeatures, feature]
                                : sessionState.selectedXFeatures.filter((f) => f.id !== feature.id);
                            updateSessionState('selectedXFeatures', updatedFeatures);
                        }}
                        selectedItems={sessionState.selectedXFeatures}
                    />
                </div>
                {/* X Features Selection */}
                <div>
                    <SelectionBox
                        label="Select y Features"
                        items={sessionState.allYFeatures}
                        itemKey="id"
                        displayText={(item) => item.name}
                        onSelectionChange={(feature, isSelected) => {
                            const updatedFeatures = isSelected
                                ? [...sessionState.selectedYFeatures, feature]
                                : sessionState.selectedYFeatures.filter((f) => f.id !== feature.id);
                            updateSessionState('selectedYFeatures', updatedFeatures);
                        }}
                        selectedItems={sessionState.selectedYFeatures}
                    />
                </div>
            </div>

            {/* Submit Button */}
            <button
                className="bg-blue-500 text-white py-2 px-4 mt-6 rounded-lg hover:bg-blue-600"
                onClick={() => {
                    handleSubmit();

                }}
            >
                Start Training Session
            </button>
        </div>
    );
}

export default TrainingConfiguration;