# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Dict, List

import pandas as pd

from constants import WALKING, CABINETS, STANDING, SITTING, SUPPORTED_ACTIVITIES
from parser.check_create_directories import check_in_path
from parser.extract_from_path import get_folder_name_from_path
from parser.save_to_csv import save_data_to_csv
from processing.filters import apply_filters
from load.load_sync_data import load_data_from_csv
from processing.task_segmentation import segment_tasks


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def processing(sync_data_path: str, output_path: str, fs: int = 100) -> None:
    """
    Processes and filters signal data from csv files in a directory structure, storing the results in a dictionary.

    This function goes through each subfolder in the given directory path, applies median and low pass filters to
    accelerometer and gyroscope data and removes the gravitational component in accelerometer data.

    Parameters:
        sync_data_path (str): The path to the directory containing folders of synchronized signal data files.
        output_path (str): Path where the data should be saved
        fs (int, optional): The sampling frequency used for the processing process. Defaults to 100 Hz.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where each key is the folder name and each value is a DataFrame
        containing the filtered data from that folder.

    """

    check_in_path(sync_data_path, '.csv')

    for folder_name in os.listdir(sync_data_path):

        folder_path = os.path.join(sync_data_path, folder_name)

        folder_name = get_folder_name_from_path(folder_path)

        for filename in os.listdir(folder_path):
            # get the path to the signals
            file_path = os.path.join(folder_path, filename)

            # load data to csv
            data = load_data_from_csv(file_path)

            # cut activities
            tasks_array = segment_tasks(folder_name, data)

            filtered_tasks = []

            for df in tasks_array:
                # filter signals
                filtered_data = apply_filters(df, fs)
                filtered_tasks.append(filtered_data)

            # generate output filenames
            output_filenames = _generate_task_filenames(folder_name, filename)

            for df, output_filename in zip(filtered_tasks, output_filenames):
                # save data to csv
                save_data_to_csv(output_filename, df, output_path, folder_name)

        # inform user
        print(f"Segment and filter{folder_name} tasks")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

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
        suffixes = ['_gestures', '_no_gestures']

    elif SITTING in folder_name:
        suffixes = ['_sit']

    else:
        raise ValueError(f"The activity: {folder_name} is not supported. "
                         f"Supported activities are {SUPPORTED_ACTIVITIES}")

    for suffix in suffixes:
        # add the suffix and extension to the previous filename
        new_filename = f"{base_filename}{suffix}{extension}"
        filenames.append(new_filename)

    return filenames
