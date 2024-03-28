# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from datetime import datetime, time

import pandas as pd

from load.load_sync_data import load_used_devices_data
from .common import crop_dataframes_on_shift, join_dataframes_on_index, generate_filename, get_folder_name_from_path, \
    sync_data_to_csv
from load.load_raw_data import load_logger_file
from typing import Dict, Tuple, Union


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

# check sync type depending on logger file if yes - sync on logger file
# if not - sync on filename timestamps


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _get_datetime_object(datetime_dic: Dict[str, Tuple[str, str]]) -> Dict[str, datetime]:
    """
    Converts separate date and time strings into a single datetime object.

    Parameters:
        datetime_dic: Dict[str, Tuple[str,str]]
            Dictionary where the keys are the device names and the values are the
            dates and times extracted from the filenames

    Returns:
        A dictionary containing datetime object representing the specified date and time as values
        and the device names as keys.

    """
    datetime_obj_dic = {}

    for device, (date, time) in datetime_dic.items():
        date = datetime.strptime(date, "%Y-%m-%d")
        time = datetime.strptime(time, "%H:%M:%S").time()

        # Combine date and time into a datetime object
        datetime_obj = datetime.combine(date, time)

        # store in dictionary with correspondent device
        datetime_obj_dic[device] = datetime_obj

    return datetime_obj_dic


# def _calculate_start_time_difference(datetime_obj_dic: Dict[str, datetime], sampling_rate: int = 100) -> int:
#     """
#     Calculates the time difference in seconds between the start times of two devices,
#     and converts this time difference to the number of samples based on the sampling rate.
#
#     Parameters:
#         datetime_obj_dic (Dict[str, datetime]): Dictionary containing datetime objects for at least two devices.
#         sampling_rate (int): The sampling rate in Hz.
#
#     Returns:
#         float: The time difference in number of samples. A positive value indicates the first
#                mentioned device started before the second, and a negative value indicates the
#                first mentioned device started after the second.
#     """
#     if len(datetime_obj_dic) != 2:
#         raise ValueError("The dictionary must contain datetime objects for exactly two devices.")
#
#     # Convert the datetime objects to a list to ensure correct ordering for comparison
#     datetimes = list(datetime_obj_dic.values())
#
#     # Calculate the seconds difference
#     seconds_difference = (datetimes[1] - datetimes[0]).total_seconds()
#
#     # The sign of seconds_difference will indicate which device started first
#
#     # Convert the seconds difference to samples
#     samples_difference = int(seconds_difference * sampling_rate)
#
#     return samples_difference
def _time_to_seconds(t: time) -> float:
    """Convert a datetime.time object to seconds."""
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000


def _calculate_start_time_difference(datetime_obj_dic: Dict[str, Union[datetime, time]],
                                     sampling_rate: int = 100) -> int:
    """
    Calculates the time difference in seconds between the start times of two devices,
    which can be either datetime objects or time objects, and converts this time difference
    to the number of samples based on the sampling rate.

    Parameters:
        datetime_obj_dic (Dict[str, Union[datetime, time]]):
        Dictionary containing datetime objects or time objects for two devices.

        sampling_rate (int):
        The sampling rate in Hz to convert the shift from seconds to samples

    Returns:
         The shift between the signals in samples. A positive value indicates the first
         mentioned device starts first, and a negative value indicates the
         first mentioned device starts last.
    """
    if len(datetime_obj_dic) != 2:
        raise ValueError("The dictionary must contain datetime or time objects for exactly two devices.")

    # Convert the datetime/time objects to a list to ensure correct ordering for comparison
    times = list(datetime_obj_dic.values())

    # Determine if dealing with datetime objects or time objects and calculate the difference accordingly
    if all(isinstance(t, datetime) for t in times):
        # For datetime objects, calculate the difference directly
        seconds_difference = (times[1] - times[0]).total_seconds()

    elif all(isinstance(t, time) for t in times):
        # For time objects, convert to seconds since midnight then calculate the difference
        seconds_since_midnight = [_time_to_seconds(t) for t in times]
        seconds_difference = seconds_since_midnight[1] - seconds_since_midnight[0]

    else:
        raise ValueError("The dictionary values must be all datetime objects or all time objects.")

    # Convert the seconds difference to samples
    samples_difference = int(seconds_difference * sampling_rate)

    return samples_difference


def sync_on_filename_timestamps(folder_path: str, output_path: str) -> None:
    # get the dataframes of the signals in the folder
    dataframes_dic, datetimes_dic = load_used_devices_data(folder_path)

    # get start times as datetime objects
    datetimes_obj_dic = _get_datetime_object(datetimes_dic)

    # calculate start time difference
    samples_difference = _calculate_start_time_difference(datetimes_obj_dic)

    # crop signals
    sync_signal_1, sync_signal_2 = crop_dataframes_on_shift(samples_difference, dataframes_dic)

    # join signals into one dataframe
    df_joined = join_dataframes_on_index(sync_signal_1, sync_signal_2)

    # get folder name
    folder_name = get_folder_name_from_path(folder_path)

    # generate file name
    output_filename = generate_filename(datetimes_dic, folder_name, sync_type="filename_timestamps")

    # save csv file
    sync_data_to_csv(output_filename, df_joined, output_path, folder_name)


def _filter_logger_file(logger_path: str) -> pd.DataFrame:
    """
    Loads a logger file and first drops rows where the logs column contains "NOISERECORDER",
    then filters it to keep only the rows where the logs column contains the text
    "SENSOR_DATA: received first data from". Additionally, strips this part of the text
    from the 'logs' column in the filtered DataFrame.

    Parameters:
    ----------
    logger_path : str
        The file path to the logger file.

    Returns:
    -------
    pd.DataFrame
        A filtered DataFrame containing only the relevant log rows, with specific
        text removed from the 'logs' column.
    """
    # Load logger file to DataFrame
    logger_df = load_logger_file(logger_path)

    # Drop rows where 'logs' column contains 'NOISERECORDER'
    logger_df = logger_df[~logger_df['logs'].str.contains("NOISERECORDER")]

    # Filter logs column to keep only the rows containing the specified text "SENSOR_DATA: received first data from"
    filtered_df = logger_df[logger_df['logs'].str.contains("SENSOR_DATA: received first data from")]

    # Strip the specified part of the string from the 'logs' column
    filtered_df['logs'] = filtered_df['logs'].str.replace("SENSOR_DATA: received first data from ", "", regex=False)

    return filtered_df


def _get_start_times_from_logger(filtered_logger_df):
    start_times_dic = {}

    # Check for 'watch' based on 'WEAR' keyword
    wear_logs = filtered_logger_df[filtered_logger_df['logs'].str.startswith('WEAR')]
    if not wear_logs.empty:
        # Convert the first matching time string to a datetime.time object
        wear_time = datetime.strptime(wear_logs.iloc[0]['time'], '%H:%M:%S.%f').time()
        start_times_dic['watch'] = wear_time

    # Iterate through the DataFrame to find 'mban' and 'phone'
    for index, row in filtered_logger_df.iterrows():

        if ':' in row['logs']:
            # Assuming this log entry represents an MBAN device
            mban_time = datetime.strptime(row['time'], '%H:%M:%S.%f').time()
            start_times_dic['mban'] = mban_time

        if 'WEAR' not in row['logs']:

            # Assuming any log entry without 'WEAR' and not identified as MBAN is from the 'phone'
            phone_time = datetime.strptime(row['time'], '%H:%M:%S.%f').time()

            if 'phone' not in start_times_dic:  # Ensure we capture the first occurrence
                start_times_dic['phone'] = phone_time

    return start_times_dic


def _get_used_devices_start_times(dataframes_dic, filtered_logger_df):
    all_start_times_dic = _get_start_times_from_logger(filtered_logger_df)

    start_times_dic = {}

    for device in dataframes_dic.keys():

        if device in all_start_times_dic.keys():
            start_times_dic[device] = all_start_times_dic[device]

    return start_times_dic
