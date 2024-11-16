import numpy as np


def remove_nan_rows(arr):
    """
    Removes rows along the X dimension from a 3D array where any NaN exists
    in the last dimension for any value along the Y dimension.

    Parameters:
    - arr: 3D NumPy array of shape (X, Y, 1)

    Returns:
    - cleaned_arr: The array with rows containing NaNs removed
    """
    # Check for NaNs in the last dimension across Y for each row in X
    nan_mask = np.isnan(arr[:, :, 0]).any(axis=1)
    # Create a mask to identify rows without NaN values
    mask = ~nan_mask
    # Use the mask to filter out rows with NaN values
    cleaned_arr = arr[mask, :, :]
    print(f"Removed {arr.shape[0] - cleaned_arr.shape[0]} rows with NaN values")
    return cleaned_arr