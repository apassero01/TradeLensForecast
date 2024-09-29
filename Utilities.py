from sklearn.preprocessing import MinMaxScaler


def scale_sequence_by_group(features, feature_dict, sequences):
    '''
    Scale the sequences by feature group
    '''

    indices = [feature_dict[feature] for feature in features]

    for i, seq in enumerate(sequences):
        scaler = MinMaxScaler(feature_range = (0, 1))

        combined_seq = seq[:, indices].reshape(-1,1)

        scaled_combined_series = scaler.fit_transform(combined_seq)

        sequences[i][:, indices] = scaled_combined_series.reshape(seq.shape[0], len(indices))

    return sequences