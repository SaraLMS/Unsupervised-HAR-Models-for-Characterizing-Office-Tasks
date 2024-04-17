# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import os
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def create_dir(path: str, folder_name: str) -> str:
    """
    creates a new directory in the specified path
    :param path: the path in which the folder_name should be created
    :param folder_name: the name of the folder that should be created
    :return: the full path to the created folder
    """

    # join path and folder
    new_path = os.path.join(path, folder_name)

    # check if the folder does not exist yet
    if not os.path.exists(new_path):
        # create the folder
        os.makedirs(new_path)

    return new_path


def sync_signals(tau: int, array_df: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronize two signals based on a given shift (tau) and adjust their lengths to match.

    This function aligns two signals by cropping the shift (tau). If tau is positive,
    it indicates that signal_1 starts tau samples after signal_2. If tau is negative, signal_2
    starts tau samples after signal_1. After adjusting for tau, the lengths of the signals are
    equalized to match the length of the shorter signal. The time axis of the cut signal is replaced
    with the time axis of the other signal and then is set as index of the DataFrames.

    Parameters:
        tau: int
            The shift in samples between the two signals.

        array_df: List[pd.Dataframe]
            Array containing the two signals from two different devices in the one folder

    Returns:
        The two signals synchronized

    """
    signal_1 = array_df[0]
    signal_2 = array_df[1]

    # Adjust for tau
    if tau > 0:
        signal_1 = signal_1.iloc[tau:].reset_index(drop=True)
    elif tau < 0:
        signal_2 = signal_2.iloc[-tau:].reset_index(drop=True)

    # Equalize lengths after tau adjustment
    min_length = min(len(signal_1), len(signal_2))
    signal_1 = signal_1.iloc[:min_length].reset_index(drop=True)
    signal_2 = signal_2.iloc[:min_length].reset_index(drop=True)

    # Synchronize time axes
    # Here we decide which time axis to keep based on which signal was cut for tau.
    if tau >= 0:
        # Keeping signal_2's time axis, as signal_1 was cut for tau
        signal_1['sec'] = signal_2['sec'].values
    else:
        # Keeping signal_1's time axis, as signal_2 was cut for tau
        signal_2['sec'] = signal_1['sec'].values

    # set time axis as index
    signal_1.set_index('sec', inplace=True)
    signal_2.set_index('sec', inplace=True)

    return signal_1, signal_2


def crop_dataframes_on_shift(tau: int, dataframes_dic: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synchronize sensor data from chosen devices.

    Parameters:
        tau: int
        The shift in samples between the two signals.

        dataframes_dic (Dict[str, pd.DataFrame]):
        Dictionary containing the chosen device names as keys and sensor data from said devices as values.

    Returns:
        The two signals synchronized

    """
    # array for holding the two signals
    array_df = []

    for device, df in dataframes_dic.items():
        array_df.append(df)

    # crop dataframes based on tau
    sync_signal_1, sync_signal_2 = sync_signals(tau, array_df)

    return sync_signal_1, sync_signal_2


def join_dataframes_on_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join two dataframes of the same lenght on their index.

    Parameters:
    - df1: pandas.DataFrame
    - df2: pandas.DataFrame

    Returns:
    - pandas.DataFrame: Joined dataframe.
    """
    df_joined = df1.join(df2, how='inner')

    return df_joined


def generate_filename(datetime_dic: Dict[str, Tuple[str, str]], folder_name: str, prefix: str, sync_type: str) -> str:
    """
    Generate a filename for synchronized data.

    Parameters:
        datetime_dic: Dict[str, Tuple[str,str]]
        Dictionary where the keys are the device names and the values are the
        dates and times extracted from the filenames.

        folder_name (str):
        The name of the folder containing the data.

        sync_type (str):
        The synchronization method used for the data.

        prefix (str):
        Prefix to be added to the filename names.

    Returns:
        str: The generated filename for the synchronized data.
    """

    # array for holding the device names
    devices = []

    # array for holding the dates in the filenames
    dates = []

    # array for holding the times in the filenames
    times = []

    for device, (date, time) in datetime_dic.items():
        devices.append(device)
        dates.append(date)
        # Replace colons with underscores
        time = datetime.strptime(time, "%H:%M:%S").strftime("%H_%M_%S")
        times.append(time)

    # Construct the output filename
    # date and time from the first file in the folder
    output_name = f"{prefix}_synchronized_{devices[0]}_{devices[1]}_{folder_name}_{dates[0]}_{times[0]}_{sync_type}.csv"

    return output_name


def save_data_to_csv(output_filename: str, signals_df: pd.DataFrame, output_path: str, folder_name: str) -> None:
    """
    Saves synchronized sensor data to a csv file.

    Parameters:
        output_filename (str):
        The name for the output file.

        signals_df (pd.DataFrame):
        DataFrame containing the synchronized sensor data to be saved to a csv file

        output_path (str):
        Location where the file should be saved.

        folder_name (str):
        The name of the folder containing the data.
    """
    # create dir
    output_path = create_dir(output_path, folder_name)

    # add filename to get full path
    output_path = os.path.join(output_path, output_filename)

    # save dataframe to csv file
    signals_df.to_csv(output_path)
