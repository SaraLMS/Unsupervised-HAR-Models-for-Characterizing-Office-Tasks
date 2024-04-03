# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import os
import pandas as pd
from typing import Dict, Tuple, Any

from synchronization.sync_parser import extract_date_time


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_used_devices_data(folder_path: str, time_in_seconds: bool = True) -> Tuple[
    dict[str, pd.DataFrame], dict[str, tuple[Any, Any]]]:
    """
    Loads sensor data from used devices.

    Parameters:
    ----------
    folder_path : str
        The directory path containing sensor data files.

    time_in_seconds: bool (default = True)
        True if time in the sensor data is in seconds.

    Returns:
    -------
    A dictionary where keys are the device names and values are the DataFrames containing the sensor data.

    """
    used_devices_dict = _get_used_devices_from_path(folder_path)

    dataframes_dict = {}
    datetimes_dic = {}

    for device, path in used_devices_dict.items():
        # load data to a pandas dataframe
        df = load_data_from_csv(path)

        # if times is in seconds change column name to 'sec
        # TODO ASK PHILLIP
        if time_in_seconds:
            df.rename(columns={'nSeq': 'sec'}, inplace=True)

        date, time = extract_date_time(path)

        dataframes_dict[device] = df

        datetimes_dic[device] = date, time

    return dataframes_dict, datetimes_dic


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Loads data to a pandas dataframe.

    Parameters:
        file_path (str):
            Path of the file to be loaded.

    Returns:
        Pandas DataFrame containing the data.

    """
    # load csv file to a pandas DataFrame
    df = pd.read_csv(file_path, index_col=0)

    return df


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_used_devices_from_path(folder_path: str) -> Dict[str, str]:
    """
    Identifies and maps supported sensor devices to their data file paths within a given folder.

    Parameters:
    ----------
    folder_path : str
        The directory path containing sensor data files.

    Returns:
    -------
    A dictionary where keys are device names and values are the full paths to their
    corresponding data files within the specified folder.

    """
    supported_devices = ['phone', 'watch', 'mban']
    used_devices_dict = {}

    for filename in os.listdir(folder_path):
        # get the path of the csv file
        data_path = os.path.join(folder_path, filename)

        # Check if the filename contains any of the supported sensors
        for device in supported_devices:
            if device in filename:
                used_devices_dict[device] = data_path
                break
        else:
            # This else clause belongs to the for-loop. It executes if the loop completes normally (no break)
            raise ValueError(f"Unsupported device type in filename: {filename}")

    return used_devices_dict
