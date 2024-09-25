# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import os
from typing import List
import re

# internal imports
import load
import parser
from constants import (WALKING, CABINETS, STANDING, SITTING, SUPPORTED_ACTIVITIES, STAIRS, CSV,
                       ACCELEROMETER_PREFIX, SUPPORTED_PREFIXES, WALKING_SUFFIXES, CABINETS_SUFFIXES, STANDING_SUFFIXES,
                       SITTING_SUFFIXES, STAIRS_4SUFFIXES, STAIRS_8SUFFIXES)
from .filters import median_and_lowpass_filter, gravitational_filter
from .task_segmentation import segment_tasks


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def process_all(main_input_path: str, output_base_path: str, device_sensors_foldername: str) -> None:
    """
    Signal pre-processing function. See more details in the processor function bellow

    :param main_input_path: str
    Path to the main folder containing subfolders with synchronized sensor data
    (i.e., ../main_folder/subfolders/sensor_data.txt). The sub folders should have the following structure:
    four characters, starting with an upper case letter, and followed by three digits (i.e., P001)

    :param output_base_path: str
    Path to the location where the file containing the synchronized data should be saved.

    :param device_sensors_foldername: str
    Name of the folder containing the loaded sensors and devices (i.e., acc_gyr_mag_phone_watch)

    :return: None
    """

    # Compile the regular expression for valid subfolder names
    pattern = re.compile(r'^[A-Z]\d{3}$')

    for sub_folder in os.listdir(main_input_path):

        if pattern.match(sub_folder):
            raw_data_in_path = os.path.join(main_input_path, sub_folder)

            for devices_sensors_folders in os.listdir(raw_data_in_path):

                if devices_sensors_folders == device_sensors_foldername:

                    # check the which devices and sensor data to load and process based on the folder name
                    raw_data_in_path = os.path.join(raw_data_in_path, devices_sensors_folders)

                    # Check if 'sync_devices' or 'sync_android_sensors' exist
                    if 'sync_devices' in os.listdir(raw_data_in_path):
                        sync_folder = 'sync_devices'
                    elif 'sync_android_sensors' in os.listdir(raw_data_in_path):
                        sync_folder = 'sync_android_sensors'
                    else:
                        # Raise an exception if neither folder is found
                        raise ValueError(
                            f"Neither 'sync_devices' nor 'sync_android_sensors' were found in {raw_data_in_path}")

                    # create the path to the folder with the synchronized data
                    sync_data_in_path = os.path.join(raw_data_in_path, sync_folder)

                    print(sync_data_in_path)
                    processor(sync_data_in_path, output_base_path, device_sensors_foldername, sub_folder)


def processor(sync_data_path: str, output_base_path: str, device_sensors_foldername: str, sub_folder: str,
              raw_folder_name: str = "raw_tasks", filtered_folder_name: str = "filtered_tasks",
              save_raw_tasks: bool = True, fs: int = 100, impulse_response_samples: int = 200) -> None:
    """
    This function goes through each subfolder in the given directory path, segments and filters the signals

    Segment different tasks within the same recording. Onset-based segmentation for walking signals and peak-based
    segmentation for sitting and standing signals.

    For filtering, applies median and low pass filters to accelerometer and gyroscope data and removes
    the gravitational component from accelerometer data.

    :param sync_data_path: str
    Path to the folder (i.e., sync_devices) containing the synchronized data main_folder/subfolder/sync_devices/sync_data.csv

    :param output_base_path: str
    Path to the base path were the raw segments and filtered segments should be saved

    :param device_sensors_foldername: str
    Name of the folder containing the loaded sensors and devices (i.e., acc_gyr_mag_phone_watch)

    :param sub_folder: str
    Name of the subfolder which contains the synchronized data main_folder/subfolder/sync_devices/sync_data.csv

    :param raw_folder_name: str (default = raw_tasks)
    Name of the folder where to store the raw signal segments

    :param filtered_folder_name: str (default = filtered_tasks)
    Name of the folder where to store the filtered signal segments

    :param save_raw_tasks: bool (default = True)
    Save the raw signal segments. False not to save

    :param fs: int
    Sampling frequency

    :param impulse_response_samples: int
    Number of samples to be cut at the start of each segment to remove the impulse response of the filters

    :return: None
    """
    # check path
    parser.check_in_path(sync_data_path, CSV)

    # add the sub folder to the base (main) path
    output_base_path = os.path.join(output_base_path, sub_folder, device_sensors_foldername)

    # create directories with the new folder names for the raw and filtered segments
    raw_output_path = parser.create_dir(output_base_path, raw_folder_name)
    filtered_output_path = parser.create_dir(output_base_path, filtered_folder_name)

    print(f"Pre-processing {sub_folder} signals")

    # iterate through the sub folders
    for folder_name in os.listdir(sync_data_path):

        folder_path = os.path.join(sync_data_path, folder_name)

        for filename in os.listdir(folder_path):
            # get the path to the signals
            file_path = os.path.join(folder_path, filename)

            # load data to csv
            data = load.load_data_from_csv(file_path)

            # cut tasks
            tasks_array = segment_tasks(folder_name, data)

            if save_raw_tasks:
                # generate output filenames
                output_filenames = _generate_task_filenames(folder_name, filename, 4)

                for df, output_filename in zip(tasks_array, output_filenames):
                    # save data to csv
                    parser.save_data_to_csv(output_filename, df, raw_output_path, folder_name)

            # list to store the segmented and filtered signals
            filtered_tasks = []

            for df in tasks_array:
                # filter signals
                filtered_data = _apply_filters(df, fs)

                # cut first 200 samples to remove impulse response from the butterworth filters
                filtered_data = filtered_data.iloc[impulse_response_samples:]
                filtered_tasks.append(filtered_data)

            # generate filename
            output_filenames = _generate_task_filenames(folder_name, filename, 4)

            # special case if it's STAIRS
            if STAIRS in folder_name:
                nr_stairs_segments = len(filtered_tasks)
                # generate output filenames
                output_filenames = _generate_task_filenames(folder_name, filename, nr_stairs_segments)

            for df, output_filename in zip(filtered_tasks, output_filenames):
                # TODO: @p-probst allow for saving in different file formats
                # save data to csv
                parser.save_data_to_csv(output_filename, df, filtered_output_path, folder_name)

        # inform user
        print(f"Segmented and filtered {folder_name} tasks")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _apply_filters(data: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Applies various filters to sensor data columns in a CSV file.

    This function processes each sensor data column in the file, applying median and lowpass filters.
    For accelerometer data, it additionally removes the gravitational component.

    :param data: pd.DataFrame
    Pandas DataFrame containing the sensor data

    :param fs: int
    Sampling frequency

    :return: pd.DataFrame
    A DataFrame containing the filtered sensor data, with the same structure as the input file.
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


def _generate_task_filenames(folder_name: str, filename: str, nr_stairs_segments: int) -> List[str]:
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

    # TODO pass suffixes as parameter check nr of segments and suffixes (I dont remember what this was about)
    # get the suffixes to be added according to the activity
    if WALKING in folder_name:
        suffixes = WALKING_SUFFIXES

    elif CABINETS in folder_name:
        suffixes = CABINETS_SUFFIXES

    elif STANDING in folder_name:
        suffixes = STANDING_SUFFIXES

    elif SITTING in folder_name:
        suffixes = SITTING_SUFFIXES

    elif STAIRS in folder_name:

        if nr_stairs_segments == 4:
            # 2 segments going up and 2 segments going down
            suffixes = STAIRS_4SUFFIXES

        elif nr_stairs_segments == 8:
            # 4 segments going up and 4 segments going down
            suffixes = STAIRS_8SUFFIXES

        else:
            raise ValueError(f"Incorrect number of stairs segments: Can only be 4 or 8.")

    else:
        raise ValueError(f"The activity: {folder_name} is not supported. "
                         f"Supported activities are {SUPPORTED_ACTIVITIES}")

    for suffix in suffixes:
        # add the suffix and extension to the previous filename
        new_filename = f"{base_filename}{suffix}{extension}"
        filenames.append(new_filename)

    return filenames
