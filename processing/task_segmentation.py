# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Union

import pandas as pd

from constants import WALKING, STANDING, SITTING, CABINETS, SUPPORTED_ACTIVITIES


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def segment_tasks(folder_name: str, data: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Segments the given DataFrame based on activity types defined in the folder name.

    :param folder_name: str.
        The name of the folder which hints at the type of activity contained within. It determines how the data should
        be segmented and processed.
    :param data: pd.DataFrame.
        The DataFrame containing sensor data to be segmented by task.

    :return: List[pd.DataFrame].
        A list of DataFrames, where each DataFrame is a segment of the original data corresponding to a specific task
        within the given activity type.
    """
    # get time back to column named 'sec'
    data = _reset_index_to_column(data)

    # dataframes array
    tasks = []

    # check type of activity
    if SITTING in folder_name:

        # cut sitting task and store in a list
        sitting_task_df = _cut_sitting_tasks(data)
        tasks.append(sitting_task_df)

    elif CABINETS in folder_name:

        # cut cabinets tasks and store in a list
        coffee_task, folder_task = _cut_cabinets_tasks(data)
        tasks += [coffee_task, folder_task]

    elif WALKING in folder_name:

        # cut walking tasks and store in a list
        walking_tasks_list = _cut_walking_tasks(data)
        tasks.extend(walking_tasks_list)

    elif STANDING in folder_name:

        # cut standing tasks and store in a list.
        standing_tasks_list = _cut_standing_tasks(data)
        tasks.extend(standing_tasks_list)

    if not any(activity in folder_name for activity in SUPPORTED_ACTIVITIES):
        # If no supported activity is found, raise a ValueError
        raise ValueError(f"The activity: {folder_name} is not supported")

    return tasks


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _reset_index_to_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset the index and convert it into a column called 'sec'.
    :param df: pd.DataFrame.
        Dataframe containing the sensor data with the index being the time in seconds.
    :return: pd.DataFrame.
        Dataframe with the previous index in seconds converted into a column 'sec' and with new index in samples.
    """

    # Move the current index to a column and reset
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'sec'}, inplace=True)  # Rename the new column to 'sec'
    return df


def _cut_recording_time(df: pd.DataFrame, recording_size: int) -> pd.DataFrame:
    """
    Cuts the data in the dataframe to keep the last samples that correspond to the recording time. Removes the initial
    part pertaining to connection and synchronization.
    :param df: pd.DataFrame.
        Dataframe containing the sensor data to be cut
    :param recording_size: int.
        Duration in samples of the acquisition without connection/synchronization and stop at the end.
    :return: pd.DataFrame.
        Dataframe cut to the recording_size
    """
    df = df.tail(recording_size)
    df.reset_index(drop=True, inplace=True)
    return df


def _trim_data(sensor_data: pd.Series, trim_length: int, print_message=False):
    """
    trims the data in emg_series to the length defined by trim_length. This is done by trimming out a window of
    size trim_length in the middle of the emg_series. The input emg_series should have an index that runs continuously
    from [0:len(emg_series)], otherwise the trimming will not be performed as expected.
    :param sensor_data: the emg_series containing the longest sitting period for a particular session
    :param trim_length: the length to which the signal should be trimmed
    :param print_message: boolean that defines whether a message for the user should be printed. Default: True
    :return: the emg_series trimmed to trim_length
    """

    # get the length of the series
    series_length = sensor_data.size

    if print_message:
        # inform user
        print("(a) trimming data to shortest sequence length: {} samples".format(trim_length))

    # check if the length of the series is smaller to the trim_length
    if series_length < trim_length:

        # inform user
        IOError("--> Error: the data_series is smaller than the trim_length.")

    # check for same length. This is the case when the shortest sitting sequence is passed
    elif series_length == trim_length:

        return sensor_data

    else:

        # calculate start and stop indices
        start_idx = (series_length - trim_length) // 2
        stop_idx = start_idx + trim_length - 1

        # trim the emg series
        trimmed_data_series = sensor_data.loc[start_idx:stop_idx]

        return trimmed_data_series.reset_index(drop=True)


def _trim_dataframe(df: pd.DataFrame, trim_length: int) -> pd.DataFrame:
    """
    Trims all the columns in a dataframe to the trim length
    :param df: pd.DataFrame.
        Dataframe containing the sensor data
    :param trim_length: int.
        The length to which the signal should be trimmed
    :return: The dataframe trimmed to the trim_length.
    """
    df_trimmed = df.copy()
    for column in df.columns:
        df_trimmed[column] = _trim_data(df_trimmed[column], trim_length)
    df_trimmed.dropna(inplace=True)
    return df_trimmed


def _cut_and_trim_task(df: pd.DataFrame, start: int, end: Union[int, None], trim_size: int) -> pd.DataFrame:
    """
    Cuts the DataFrame from 'start' to 'end', resets the index, and trims.
    :param df: pd.DataFrame.
        Dataframe containing the sensor data.
    :param start: int.
         The starting index in samples from which to begin the cut.
    :param: end int or None.
        The ending index in samples from which to stop the cut. If 'None', cuts until the end of the dataframe.
    :param: trim_size int.
        Number of samples to trim after cutting.
    :return: pd.DataFrame
        The dataframe cut and trimmed.
    """

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


def _cut_sitting_tasks(df: pd.DataFrame, recording_size: int = 270000, trim_length: int = 261000,
                       end_samples: int = 1500) -> pd.DataFrame:
    """
    Cuts the specified DataFrame to retain only the essential part of the sitting task recordings,
    removes synchronization and connection-related data, and trims to the desired length. recording_size,
    trim_size and end_samples should be chosen according to the duration of the acquisition.

    Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. If sitting was done all
    in one go, the trim_length should still be n * 87 000 samples to guarantee the same duration to all the
    activities performed.

    :param df: pd.DataFrame.
        Dataframe containing the sensor data.
    :param recording_size: int.
        Duration in samples of the acquisition without connection/synchronization and stop at the end.
    :param trim_length: int.
        The length to which the signal should be trimmed.
    :param end_samples:
        Duration in samples of the stop before the end of the recording.
    :return: pd.DataFrame.
        Dataframe containing the sitting task only.
    """
    # cut end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # trim the data
    df = _trim_dataframe(df, trim_length)

    return df


def _cut_cabinets_tasks(df: pd.DataFrame, recording_size: int = 91000, stop_size: int = 1000,
                        end_samples: int = 1500) -> List[pd.DataFrame]:
    """
    Cuts the specified DataFrame to retain only the tasks performed during cabinets recordings,
    removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
    length. Segments the two cabinets tasks in this order: coffee and folders. recording_size, trim_size and
    end_samples should be chosen according to the duration of the acquisition.

    Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
    have the same duration to guarantee that all activities are equally represented.

    :param df: pd.DataFrame.
        Dataframe containing the sensor data.
    :param recording_size: int.
        Duration in samples of the acquisition without connection/synchronization and stop at the end.
    :param stop_size: int.
        Duration in samples of the stops in between tasks.
    :param end_samples:
        Duration in samples of the stop before the end of the recording.
    :return: List[pd.DataFrame].
        List with two dataframes: the coffee task and folder task.
    """
    # Cut the end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # Get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # Remove 10 seconds stop separating coffee and folder tasks
    # First segment to keep
    coffee_task_df = _cut_and_trim_task(df, 0, int(recording_size / 2 - stop_size/2), 1500)

    # Second segment to keep
    folder_task_df = _cut_and_trim_task(df, int(recording_size / 2 + stop_size/2), None, 1500)

    # Store the DataFrames in a list and return
    return [coffee_task_df, folder_task_df]


def _cut_walking_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
                       end_samples: int = 1500, segment_trim_size: int = 1000) -> List[pd.DataFrame]:
    """
    Cuts the specified DataFrame to retain only the tasks performed during walking recordings,
    removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
    length. Segments the three walking tasks/speeds in this order: slow, medium, and fast. recording_size, trim_size and
    end_samples should be chosen according to the duration of the acquisition.

    Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
    have the same duration to guarantee that all activities are equally represented.

    :param df: pd.DataFrame.
        Dataframe containing the sensor data.
    :param recording_size: int.
        Duration in samples of the acquisition without connection/synchronization and stop at the end.
    :param stop_size: int.
        Duration in samples of the stops in between tasks.
    :param end_samples:
        Duration in samples of the stop before the end of the recording.
    :return: List[pd.DataFrame].
        List with three dataframes: slow, medium and fast walking speeds.
    """
    # Cut the end part
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)

    # Get only the actual recording, removes connection and sync parts
    df = _cut_recording_time(df, recording_size)

    # Calculate recording size excluding stops
    recording_size_no_stops = int(recording_size - 2 * stop_size)

    # cut first walking task - slow walking speed
    walk_slow_df = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 3), segment_trim_size)

    # cut second walking task - medium walking speed
    walk_medium_df = _cut_and_trim_task(df, int(recording_size_no_stops / 3 + stop_size),
                                        int(2 * recording_size_no_stops / 3 + stop_size), segment_trim_size)

    # cut third walking task - fast walking speed
    walk_fast_df = _cut_and_trim_task(df, int(2 * recording_size_no_stops / 3 + 2 * stop_size),
                                      None, segment_trim_size)  # Going to the end of the DataFrame

    # Store the dataframes in a list
    tasks_list = [walk_slow_df, walk_medium_df, walk_fast_df]

    return tasks_list


def _cut_standing_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
                        end_samples: int = 1500, segment_trim_size: int = 1500) -> List[pd.DataFrame]:
    """
    Cuts the specified DataFrame to retain only the tasks performed during standing recordings,
    removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
    length. Segments the two standing tasks in this order: standing with gestures and standing without gestures.
    recording_size, trim_size and end_samples should be chosen according to the duration of the acquisition.

    Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
    have the same duration to guarantee that all activities are equally represented.

    :param df: pd.DataFrame.
        Dataframe containing the sensor data.
    :param recording_size: int.
        Duration in samples of the acquisition without connection/synchronization and stop at the end.
    :param stop_size: int.
        Duration in samples of the stops in between tasks.
    :param end_samples:
        Duration in samples of the stop before the end of the recording.
    :return: List[pd.DataFrame].
        List with two dataframes: standing with and without gestures.
    """
    # cut end part and get only the actual recording
    df = df.iloc[:-end_samples]
    df.reset_index(drop=True, inplace=True)
    df = _cut_recording_time(df, recording_size)

    # cut standing  where there were stops in the acquisition
    recording_size_no_stops = int(recording_size - 2 * stop_size)

    # cut first segment
    standing_no_gestures_df1 = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 4), int(segment_trim_size / 2))

    # cut second segment
    standing_with_gestures_df = _cut_and_trim_task(df, int(recording_size_no_stops / 4 + stop_size),
                                                   int(3 * recording_size_no_stops / 4 + stop_size), segment_trim_size)

    # cut third segment
    standing_no_gestures_df2 = _cut_and_trim_task(df, int(3 * recording_size_no_stops / 4 + 2 * stop_size), None,
                                                  int(segment_trim_size / 2))

    # join the separated task into one df
    standing_no_gestures_df = pd.concat([standing_no_gestures_df1, standing_no_gestures_df2], ignore_index=True)

    # store tasks
    tasks_list = [standing_with_gestures_df, standing_no_gestures_df]

    return tasks_list
