class ModelSetService:

    @staticmethod
    def serialize_model_set_state(model_set):
        model_set_state = {
            'X': model_set.X.shape if model_set.X is not None else None,
            'y': model_set.y.shape if model_set.y is not None else None,
            'X_train': model_set.X_train.shape if model_set.X_train is not None else None,
            'X_test': model_set.X_test.shape if model_set.X_test is not None else None,
            'y_train': model_set.y_train.shape if model_set.y_train is not None else None,
            'y_test': model_set.y_test.shape if model_set.y_test is not None else None,
            'X_train_scaled': model_set.X_train_scaled.shape if model_set.X_train_scaled is not None else None,
            'X_test_scaled': model_set.X_test_scaled.shape if model_set.X_test_scaled is not None else None,
            'y_train_scaled': model_set.y_train_scaled.shape if model_set.y_train_scaled is not None else None,
            'y_test_scaled': model_set.y_test_scaled.shape if model_set.y_test_scaled is not None else None,
        }
        print(model_set_state)
        return model_set_state

    @staticmethod
    def get_sequence_set_members_by_id(model_set, member_ids):
        '''
        Get a data set member by id for sequence sets NOTE: Only works for models sets with data_set type sequence set
        sequence set has a list of sequences with meta data to collect the sequences
        '''
        sequence_set = model_set.data_set
        sequences = sequence_set.sequences
        sequences_filtered = [sequence for sequence in sequences if sequence.id in member_ids]
        return sequence_set, sequences_filtered

    @staticmethod
    def get_sequence_set_metadata_by_id(model_set, member_ids):
        '''
        Get a data set member by id for sequence sets NOTE: Only works for models sets with data_set type sequence set
        sequence set has a list of sequences with meta data to collect the sequences
        '''
        sequence_set, sequences_filtered = ModelSetService.get_sequence_set_members_by_id(model_set, member_ids)
        sequence_meta_data = []
        for sequence in sequences_filtered:
            meta_data = {}
            meta_data['id'] = sequence.id
            meta_data['start_timestamp'] = sequence.start_timestamp
            meta_data['end_timestamp'] = sequence.end_timestamp
            meta_data['metadata'] = sequence_set.metadata
            sequence_meta_data.append(meta_data)

        return sequence_set, sequence_meta_data
