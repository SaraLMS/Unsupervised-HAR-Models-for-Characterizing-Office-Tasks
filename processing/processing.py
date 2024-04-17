# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict

import pandas as pd

from constants import ACCELEROMETER_PREFIX, GYROSCOPE_PREFIX
from parser.check_files_directories import check_in_path
from parser.extract_from_path import get_folder_name_from_path
from processing.filters import median_and_lowpass_filter, gravitational_filter
from load.load_sync_data import load_data_from_csv
from processing.task_segmentation import segment_tasks


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def processing(sync_data_path: str, fs: int = 100) -> Dict[str, pd.DataFrame]:
    """
    Processes and filters signal data from csv files in a directory structure, storing the results in a dictionary.

    This function goes through each subfolder in the given directory path, applies median and low pass filters to
    accelerometer and gyroscope data and removes the gravitational component in accelerometer data.

    Parameters:
        sync_data_path (str): The path to the directory containing folders of synchronized signal data files.
        fs (int, optional): The sampling frequency used for the processing process. Defaults to 100 Hz.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where each key is the folder name and each value is a DataFrame
        containing the filtered data from that folder.

    """
    # TODO check directories and csv files

    filtered_signals_dict = {}

    for folder_name in os.listdir(sync_data_path):

        folder_path = os.path.join(sync_data_path, folder_name)

        folder_name = get_folder_name_from_path(folder_path)

        for filename in os.listdir(folder_path):

            # get the path to the signals
            file_path = os.path.join(folder_path, filename)

            # filter signals
            filtered_data = _apply_filters(file_path, fs)

            # cut activities
            cut_data = segment_tasks(folder_name, filtered_data)

            # save in dictionary
            filtered_signals_dict[folder_name] = cut_data

    return filtered_signals_dict


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
