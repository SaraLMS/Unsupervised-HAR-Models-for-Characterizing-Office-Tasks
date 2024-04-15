# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

from constants import ACCELEROMETER_PREFIX, GYROSCOPE_PREFIX
from filtering.filters import median_and_lowpass_filter, gravitational_filter
from load.load_sync_data import load_data_from_csv


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _apply_filters(file_path: str, fs: int) -> pd.DataFrame:
    """
    Applies various filters to sensor data columns in a CSV file.

    This function processes each sensor data column in the file, applying median and lowpass filters.
    For accelerometer data, it additionally removes the gravitational component.

    Parameters:
        file_path (str): The file path of the CSV containing sensor data.
        fs (int): The sampling frequency of the sensor data.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered sensor data, with the same structure as the input file.
    """
    data = load_data_from_csv(file_path)

    filtered_data = data.copy()

    # Process each sensor column directly
    for sensor in filtered_data.columns:

        # Determine if the sensor is an accelerometer or gyroscope by its prefix
        if ACCELEROMETER_PREFIX in sensor or GYROSCOPE_PREFIX in sensor:
            # Get raw sensor data
            raw_data = filtered_data[sensor].values

            # Apply median and lowpass filters
            filtered_median_lowpass_data = median_and_lowpass_filter(raw_data, fs)

            if ACCELEROMETER_PREFIX in sensor:
                # For accelerometer data, additionally remove the gravitational component
                gravitational_component = gravitational_filter(raw_data, fs)

                # Remove gravitational component from filtered data
                filtered_median_lowpass_data -= gravitational_component

            # Update DataFrame with filtered sensor data
            filtered_data[sensor] = pd.Series(filtered_median_lowpass_data, index=filtered_data.index)

    return filtered_data
