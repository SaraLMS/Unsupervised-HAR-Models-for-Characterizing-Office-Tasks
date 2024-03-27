# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import os
import pandas as pd
from typing import Dict, Tuple, Any

from pandas import DataFrame

from synchronization.sync_parser import extract_date_time


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_used_devices_data(folder_path: str) -> Tuple[dict[str, pd.DataFrame], dict[str, tuple[Any, Any]]]:
    used_devices_dict = _get_used_devices_from_path(folder_path)

    dataframes_dict = {}
    datetimes_dic = {}

    for device, path in used_devices_dict.items():
        # load data to a pandas dataframe
        df = _load_data_from_csv(path)

        date, time = extract_date_time(path)

        dataframes_dict[device] = df

        datetimes_dic[device] = date, time

    return dataframes_dict, datetimes_dic


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
    A dictionary where keys are sensor types and values are the full paths to their
    corresponding data files within the specified folder.

    """
    supported_sensors = ['phone', 'watch', 'mban']
    used_devices_dict = {}

    for filename in os.listdir(folder_path):
        # get the path of the csv file
        data_path = os.path.join(folder_path, filename)

        # Check if the filename contains any of the supported sensors
        for sensor in supported_sensors:
            if sensor in filename:
                used_devices_dict[sensor] = data_path
                break
        else:
            # This else clause belongs to the for-loop. It executes if the loop completes normally (no break)
            raise ValueError(f"Unsupported sensor type in filename: {filename}")

    return used_devices_dict


def _load_data_from_csv(file_path: str) -> pd.DataFrame:
    # load csv file to a pandas DataFrame
    df = pd.read_csv(file_path, index_col=0)

    return df
