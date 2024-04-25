# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import Tuple, List

import numpy as np
import pandas as pd

from constants import WALKING, STANDING, SITTING, CABINETS, SUPPORTED_ACTIVITIES


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def segment_tasks(folder_name: str, data: pd.DataFrame):
    # get time back to column named 'sec'
    data = _reset_index_to_column(data)

    # dataframes array
    tasks = []

    # check type of activity
    if SITTING in folder_name:
        # cut
        sitting_task_df = _cut_sitting_tasks(data)
        tasks.append(sitting_task_df)

    elif CABINETS in folder_name:

        # cut cabinets tasks and save in array
        coffee_task, folder_task = _cut_cabinets_tasks(data)
        tasks += [coffee_task, folder_task]

    elif WALKING in folder_name:

        # cut walking tasks and save in array
        walking_tasks_list = _cut_walking_tasks(data)
        tasks.extend(walking_tasks_list)

    elif STANDING in folder_name:

        # cut standing tasks and save in array
        standing_tasks_list = _cut_standing_tasks(data)
        tasks.extend(standing_tasks_list)

    if not any(activity in folder_name for activity in SUPPORTED_ACTIVITIES):
        # If no supported activity is found, raise a ValueError
        raise ValueError(f"The activity: {folder_name} is not supported")

    # reset time

    # new_time_axis = _generate_time_axis(df.iloc[:, 0].values)
    # df.['sec'] = new_time_axis

    return tasks


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _reset_index_to_column(df):
    """ Reset the index and convert it into a column called 'sec'. """
    df.reset_index(inplace=True)  # This moves the index to a column
    df.rename(columns={'index': 'sec'}, inplace=True)  # Rename the new column to 'sec'
    return df


def _generate_time_axis(signal, sampling_rate=100):
    # get the number of samples
    num_samples = len(signal)

    # calculate the end of the signal in seconds
    end_time = num_samples / sampling_rate

    # generate the time axis
    time_axis = np.arange(0, end_time, 1 / sampling_rate)

    return time_axis


def _cut_recording_time(df: pd.DataFrame, recording_size: int) -> pd.DataFrame:
    """ Returns the last n samples of a DataFrame. """
    df = df.tail(recording_size)
    df.reset_index(drop=True, inplace=True)
    return df


def _trim_data(emg_series: pd.Series, trim_length: int, print_message=False):
    """
    trims the data in emg_series to the length defined by trim_length. This is done by trimming out a window of
    size trim_length in the middle of the emg_series. The input emg_series should have an index that runs continuously
    from [0:len(emg_series)], otherwise the trimming will not be performed as expected.
    :param emg_series: the emg_series containing the longest sitting period for a particular session
    :param trim_length: the length to which the signal should be trimmed
    :param print_message: boolean that defines whether a message for the user should be printed. Default: True
    :return: the emg_series trimmed to trim_length
    """

    # get the length of the series
    series_length = emg_series.size

    if print_message:
        # inform user
        print("(a) trimming data to shortest sequence length: {} samples".format(trim_length))

    # check if the length of the series is smaller to the trim_length
    if series_length < trim_length:

        # inform user
        IOError("--> Error: the emg_series is smaller than the trim_length.")

    # check for same length. This is the case when the shortest sitting sequence is passed
    elif series_length == trim_length:

        return emg_series

    else:

        # calculate start and stop indices
        start_idx = (series_length - trim_length) // 2
        stop_idx = start_idx + trim_length - 1

        # trim the emg series
        trimmed_emg_series = emg_series.loc[start_idx:stop_idx]

        return trimmed_emg_series.reset_index(drop=True)


def _trim_dataframe(df, trim_lenght):
    for column in df.columns:
        df[column] = _trim_data(df[column], trim_lenght)
    df.dropna(inplace=True)
    return df


def _cut_and_trim_task(df, start, end, trim_size):
    """Cuts the DataFrame from start to end, resets the index, and trims the length."""

    # Validate start index
    if not 0 <= start < len(df):
        raise ValueError(f"Start index {start} out of range for DataFrame of length {len(df)}.")

    # cut the part of the df containing the specific task
    # Validate end index if it's not None
    if end is not None:
        if not start < end <= len(df):
            raise ValueError(
                f"End index {end} out of range for DataFrame starting at index {start} with length {len(df)}.")
        task_df = df.iloc[start:end]

    else:
        task_df = df.iloc[start:]

    # reset index after cutting
    task_df.reset_index(drop=True, inplace=True)

    # trim the new df
    trim_length = len(task_df) - trim_size

    # Validate trim length
    if trim_length < 0 or trim_length > len(task_df):
        raise ValueError(f"Trim size {trim_size} is invalid for segment of length {len(task_df)}.")

    task_df = _trim_dataframe(task_df, trim_length)

    return task_df


def _cut_sitting_tasks(df: pd.DataFrame, recording_size: int = 90000, trim_size: int = 87000,
                       end_samples: int = 500) -> pd.DataFrame:
    # cut end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # trim the data
    df = _trim_dataframe(df, trim_size)

    return df


def _cut_cabinets_tasks(df: pd.DataFrame, recording_size: int = 91000, stop_size: int = 1000,
                        end_samples: int = 1500) -> List[pd.DataFrame]:
    # Cut the end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # Get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # Remove 10 seconds stop separating coffee and folder tasks
    # First segment to keep
    coffee_task_df = _cut_and_trim_task(df, 0, int(recording_size / 2), 1500)

    # Second segment to keep
    folder_task_df = _cut_and_trim_task(df, int(recording_size / 2 + stop_size), None, 1500)

    # Store the DataFrames in a list and return
    return [coffee_task_df, folder_task_df]


def _cut_walking_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
                       end_samples: int = 1500, segment_trim_size: int = 1000) -> List[pd.DataFrame]:
    # Cut the end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # Get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # Calculate recording size excluding stops
    recording_size_no_stops = int(recording_size - 2 * stop_size)

    # cut first walking task - slow walk
    walk_slow_df = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 3), segment_trim_size)

    # cut second walking task - medium walk
    walk_medium_df = _cut_and_trim_task(df, int(recording_size_no_stops / 3 + stop_size),
                                        int(2 * recording_size_no_stops / 3 + stop_size), segment_trim_size)

    # cut third walking task - fast walk
    walk_fast_df = _cut_and_trim_task(df, int(2 * recording_size_no_stops / 3 + 2 * stop_size),
                                      None, segment_trim_size)  # Going to the end of the DataFrame

    # Store the dataframes in a list
    tasks_list = [walk_slow_df, walk_medium_df, walk_fast_df]

    return tasks_list


def _cut_standing_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
                        end_samples: int = 1500, segment_trim_size: int = 1500) -> List[pd.DataFrame]:
    # cut end part and get only the actual recording
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)
    df = _cut_recording_time(df, recording_size)

    # cut standing  where there were stops in the acquisition
    recording_size_no_stops = int(recording_size - 2 * stop_size)

    # cut first segment
    standing_no_gestures_df1 = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 4), segment_trim_size / 2)

    # cut second segment
    standing_with_gestures_df = _cut_and_trim_task(df, int(recording_size_no_stops / 4 + stop_size),
                                                   int(3 * recording_size_no_stops / 4 + stop_size), segment_trim_size)

    # cut third segment
    standing_no_gestures_df2 = _cut_and_trim_task(df, int(3 * recording_size_no_stops / 4 + 2 * stop_size), None,
                                                  segment_trim_size / 2)

    # join the separated task into one df
    standing_no_gestures_df = pd.concat([standing_no_gestures_df1, standing_no_gestures_df2], ignore_index=True)

    # store tasks
    tasks_list = [standing_with_gestures_df, standing_no_gestures_df]

    return tasks_list
