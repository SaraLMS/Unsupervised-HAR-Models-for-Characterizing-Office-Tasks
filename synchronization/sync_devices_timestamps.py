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

from .sync_parser import check_logger_file


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def sync_timestamps(logger_folder_path: str, folder_path: str, output_path: str) -> None:
    # check if logger file exists and if it is not empty
    if check_logger_file(logger_folder_path):
        # sync signals based on logger timestamps
        _sync_on_logger_timestamps(logger_folder_path, folder_path, output_path)

    else:
        # sync signals based on filename timestamps
        _sync_on_filename_timestamps(folder_path, output_path)


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


def _calculate_start_time_difference(start_times_dic: Dict[str, Union[datetime, time]],
                                     sampling_rate: int = 100) -> int:
    """
    Calculates the time difference in seconds between the start times of two devices,
    which can be either datetime objects or time objects, and converts this time difference
    to the number of samples based on the sampling rate.

    Parameters:
        start_times_dic (Dict[str, Union[datetime, time]]):
        Dictionary containing datetime objects or time objects for two devices.

        sampling_rate (int):
        The sampling rate in Hz to convert the shift from seconds to samples

    Returns:
         The shift between the signals in samples. A positive value indicates the first
         mentioned device starts first, and a negative value indicates the
         first mentioned device starts last.
    """
    if len(start_times_dic) != 2:
        raise ValueError("The dictionary must contain datetime or time objects for exactly two devices.")

    # Convert the datetime/time objects to a list to ensure correct ordering for comparison
    times = list(start_times_dic.values())

    # Determine if dealing with datetime objects or time objects and calculate the difference accordingly
    if all(isinstance(t, datetime) for t in times):
        # For datetime objects, calculate the difference directly
        seconds_difference = (times[1] - times[0]).total_seconds()

    elif all(isinstance(t, time) for t in times):
        # For time objects, convert to seconds since midnight then calculate the difference
        seconds = [_time_to_seconds(t) for t in times]
        seconds_difference = seconds[1] - seconds[0]

    else:
        raise ValueError("The dictionary values must be all datetime objects or all time objects.")

    # Convert the seconds difference to samples
    samples_difference = int(seconds_difference * sampling_rate)

    return samples_difference


def _sync_on_filename_timestamps(folder_path: str, output_path: str) -> None:
    # get the dataframes of the signals in the folder
    dataframes_dic, datetimes_dic = load_used_devices_data(folder_path)

    # get start times as datetime objects for each device
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


def _filter_logger_file(raw_folder_path: str) -> pd.DataFrame:
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
    logger_df = load_logger_file(raw_folder_path)

    # Drop rows where 'logs' column contains 'NOISERECORDER'
    logger_df = logger_df[~logger_df['logs'].str.contains("NOISERECORDER")]

    # Filter logs column to keep only the rows containing the specified text "SENSOR_DATA: received first data from"
    filtered_df = logger_df[logger_df['logs'].str.contains("SENSOR_DATA: received first data from")]

    # Strip the specified part of the string from the 'logs' column
    filtered_df.loc[:, 'logs'] = filtered_df['logs'].str.replace("SENSOR_DATA: received first data from", "",
                                                                 regex=False)

    return filtered_df


def _get_start_times_from_logger(filtered_logger_df: pd.DataFrame) -> Dict[str, datetime.time]:
    start_times_dic = {}

    # Flags to check if the start time for each device type has been found
    found_wear = False
    found_mban = False
    found_phone = False

    # Iterate through the DataFrame to find devices
    for index, row in filtered_logger_df.iterrows():

        # Check for 'WEAR_' prefix to identify watch logs
        if not found_wear and 'WEAR' in row['logs']:
            wear_time = datetime.strptime(row['time'], '%H:%M:%S.%f').time()
            start_times_dic['watch'] = wear_time
            found_wear = True

        # Check for logs indicating MBAN devices, identified by presence of colon ':'
        elif not found_mban and ':' in row['logs']:
            mban_time = datetime.strptime(row['time'], '%H:%M:%S.%f').time()
            start_times_dic['mban'] = mban_time
            found_mban = True

        # If the log doesn't fall into previous categories, assume it's from the 'phone'
        # This is the fallback case and doesn't have a specific check
        elif not found_phone:
            phone_time = datetime.strptime(row['time'], '%H:%M:%S.%f').time()
            start_times_dic['phone'] = phone_time
            found_phone = True

        # Once all device types have been found, break out of the loop early
        if found_wear and found_mban and found_phone:
            break

    return start_times_dic


def _get_used_devices_start_times_from_logger(dataframes_dic: Dict[str, pd.DataFrame],
                                              filtered_logger_df: pd.DataFrame) -> Dict[str, time]:
    all_start_times_dic = _get_start_times_from_logger(filtered_logger_df)

    start_times_dic = {}

    for device in dataframes_dic.keys():

        if device in all_start_times_dic.keys():
            start_times_dic[device] = all_start_times_dic[device]

    return start_times_dic


def _sync_on_logger_timestamps(logger_folder_path: str, folder_path: str, output_path: str) -> None:
    # get the dataframes of the signals in the folder
    dataframes_dic, datetimes_dic = load_used_devices_data(folder_path)

    # get logger file from folder containing the raw signals
    logger_df = _filter_logger_file(logger_folder_path)

    # get start times from the devices used
    start_times_dic = _get_used_devices_start_times_from_logger(dataframes_dic, logger_df)

    # calculate start time difference between devices in samples
    shift = _calculate_start_time_difference(start_times_dic)

    # cut signals
    sync_signal_1, sync_signal_2 = crop_dataframes_on_shift(shift, dataframes_dic)

    # join signals into one dataframe
    df_joined = join_dataframes_on_index(sync_signal_1, sync_signal_2)

    # get folder name
    folder_name = get_folder_name_from_path(folder_path)

    # generate file name
    output_filename = generate_filename(datetimes_dic, folder_name, sync_type="logger_timestamps")

    # save csv file
    sync_data_to_csv(output_filename, df_joined, output_path, folder_name)
