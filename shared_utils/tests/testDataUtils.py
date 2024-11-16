from django.test import TestCase
import numpy as np

from shared_utils.utils.DataUtils import remove_nan_rows


class RemoveNanRowsTest(TestCase):
    def test_remove_nan_rows(self):
        # Define dimensions
        X = 5  # Number of rows along X dimension
        Y = 3  # Number of elements along Y dimension

        # Test 1: Rows with all NaNs
        arr = np.random.rand(X, Y, 1)
        arr[0, :, 0] = np.nan  # Entire row has NaNs
        arr[1, :, 0] = np.nan  # Entire row has NaNs

        # Expected number of rows after removal
        expected_rows_all_nan = X - 2

        # Call the function
        cleaned_arr = remove_nan_rows(arr)

        # Assertions for all NaNs in rows
        self.assertEqual(cleaned_arr.shape[0], expected_rows_all_nan, "Incorrect number of rows after NaN removal for all NaNs in rows")
        self.assertFalse(np.isnan(cleaned_arr).any(), "NaN values still present in cleaned array after all NaN rows removal")

        # Test 2: Rows with partial NaNs
        # Create a new array with a single NaN in one element of the row
        arr_partial_nan = np.random.rand(X, Y, 1)
        arr_partial_nan[0, 1, 0] = np.nan  # Only one element is NaN in the row
        arr_partial_nan[2, 0, 0] = np.nan  # Only one element is NaN in another row

        # Expected number of rows after removal (removes rows 0 and 2)
        expected_rows_partial_nan = X - 2

        # Call the function for partial NaNs
        cleaned_arr_partial_nan = remove_nan_rows(arr_partial_nan)

        # Assertions for rows with partial NaNs
        self.assertEqual(cleaned_arr_partial_nan.shape[0], expected_rows_partial_nan, "Incorrect number of rows after NaN removal for partial NaNs in rows")
        self.assertFalse(np.isnan(cleaned_arr_partial_nan).any(), "NaN values still present in cleaned array after partial NaN rows removal")

        # Optional: Print statements to confirm the test (useful for debugging)
        print(f"Original array shape with all NaNs: {arr.shape}")
        print(f"Cleaned array shape with all NaNs removed: {cleaned_arr.shape}")
        print(f"Original array shape with partial NaNs: {arr_partial_nan.shape}")
        print(f"Cleaned array shape with partial NaNs removed: {cleaned_arr_partial_nan.shape}")