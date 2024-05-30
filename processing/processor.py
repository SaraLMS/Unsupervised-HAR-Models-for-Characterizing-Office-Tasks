# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict, List

import pandas as pd

from constants import WALKING, CABINETS, STANDING, SITTING, SUPPORTED_ACTIVITIES, STAIRS
from parser.check_create_directories import check_in_path, create_dir
from parser.save_to_csv import save_data_to_csv
from processing.filters import median_and_lowpass_filter, gravitational_filter
from load.load_sync_data import load_data_from_csv
from processing.task_segmentation import segment_tasks

from constants import ACCELEROMETER_PREFIX, SUPPORTED_PREFIXES


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def processing(sync_data_path: str, output_path: str, raw_folder_name: str, filtered_folder_name: str, save_raw_tasks:bool = True, fs: int = 100) -> None:
    """
    Processes and filters signal data from csv files in a directory structure,
     storing the results in a dictionary.

    This function goes through each subfolder in the given directory path, applies median and low pass filters to
    accelerometer and gyroscope data and removes the gravitational component in accelerometer data.

    Parameters:
        sync_data_path (str): The path to the directory containing folders of synchronized signal data files.
        filtered_output_path (str): Path where the data should be saved
        fs (int, optional): The sampling frequency used for the processing process. Defaults to 100 Hz.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where each key is the folder name and each value is a DataFrame
        containing the filtered data from that folder.

    """

    check_in_path(sync_data_path, '.csv')

    # create output paths
    raw_output_path = create_dir(output_path, raw_folder_name)
    filtered_output_path = create_dir(output_path, filtered_folder_name)


    for folder_name in os.listdir(sync_data_path):

        folder_path = os.path.join(sync_data_path, folder_name)

        # removed - folder_name = get_folder_name_from_path(folder_path)

        for filename in os.listdir(folder_path):
            # get the path to the signals
            file_path = os.path.join(folder_path, filename)

            # load data to csv
            data = load_data_from_csv(file_path)

            # cut tasks
            tasks_array = segment_tasks(folder_name, data)

            if save_raw_tasks:
                # generate output filenames
                output_filenames = _generate_task_filenames(folder_name, filename)

                for df, output_filename in zip(tasks_array, output_filenames):
                    # save data to csv
                    save_data_to_csv(output_filename, df, raw_output_path, folder_name)

            # list to store the segmented and filtered signals
            filtered_tasks = []

            for df in tasks_array:
                # filter signals
                filtered_data = _apply_filters(df, fs)

                # cut first 200 samples to remove impulse response from the butterworth filters
                filtered_data = filtered_data.iloc[200:]
                filtered_tasks.append(filtered_data)

            # generate output filenames
            output_filenames = _generate_task_filenames(folder_name, filename)

            for df, output_filename in zip(filtered_tasks, output_filenames):
                # save data to csv
                save_data_to_csv(output_filename, df, filtered_output_path, folder_name)

        # inform user
        print(f"Segment and filter{folder_name} tasks")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _apply_filters(data: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Applies various filters to sensor data columns in a CSV file.

    This function processes each sensor data column in the file, applying median and lowpass filters.
    For accelerometer data, it additionally removes the gravitational component.

    Parameters:
        data (pd.DataFrame): DataFrame containing the sensor data.
        fs (int): The sampling frequency of the sensor data.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered sensor data, with the same structure as the input file.
    """

    filtered_data = data.copy()

    # Process each sensor column directly
    for sensor in filtered_data.columns:

        # Determine if the sensor is an accelerometer or gyroscope by its prefix
        if any(prefix in sensor for prefix in SUPPORTED_PREFIXES):
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


def _generate_task_filenames(folder_name: str, filename: str) -> List[str]:
    """
    Generates a list of new filenames based on the activity type specified in the folder name by appending relevant
    suffixes to the original filename.

    :param folder_name: str.
        Name of the folder indicating the activity type which will determine the suffixes added to the filename.
    :param filename: str.
        Original filename to which the suffixes will be added.
    :return: List[str].
        A list of modified filenames with activity-specific suffixes appended to the base filename.
    """
    # list to store the new filenames
    filenames = []

    # split .csv from the filename
    base_filename, extension = os.path.splitext(filename)

    # get the suffixes to be added according to the activity
    if WALKING in folder_name:
        suffixes = ['_slow', '_medium', '_fast']

    elif CABINETS in folder_name:
        suffixes = ['_coffee', '_folders']

    elif STANDING in folder_name:
        suffixes = ['_gestures', '_stand_still']

    elif SITTING in folder_name:
        suffixes = ['_sit']

    elif STAIRS in folder_name:
        suffixes = ['_stairs_up', '_stairs_down']

    else:
        raise ValueError(f"The activity: {folder_name} is not supported. "
                         f"Supported activities are {SUPPORTED_ACTIVITIES}")

    for suffix in suffixes:
        # add the suffix and extension to the previous filename
        new_filename = f"{base_filename}{suffix}{extension}"
        filenames.append(new_filename)

    return filenames
