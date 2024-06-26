# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Tuple
import numpy as np
import pandas as pd

from constants import WALKING, STANDING, SITTING, CABINETS, SUPPORTED_ACTIVITIES, STAIRS
from processing.filters import median_and_lowpass_filter, gravitational_filter, get_envelope
from scipy.signal import find_peaks


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def segment_tasks(folder_name: str, data: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Segments the given DataFrame based on activity types defined in the folder name. The trim values differ
    since some activities are more segmented and therefor the trim value is lower not to lose too much signal.
    The accelerometer from the smartphone must be selected for this method to work correctly.

    :param folder_name: str.
        The name of the folder which hints at the type of activity contained within. It determines how the data should
        be segmented and processed.
    :param data: pd.DataFrame.
        The DataFrame containing sensor data to be segmented by task.

    :return: List[pd.DataFrame].
        A list of DataFrames, where each DataFrame is a segment of the original data corresponding to a specific task
        within the given activity type.
    """
    # # get time back to column named 'sec'
    # data = _reset_index_to_column(data)
    # get the y axis from the acc
    # TODO ADD CHECK ACC from phone NEEDS TO BE CHOSEN - FOR GOOD RESULTS!!!!!!!!!!!!!!!!!!!!
    acc_series = data['yAcc'].to_numpy()
    # dataframes array
    tasks = []

    # check type of activity
    if SITTING in folder_name:

        # get starts and stops for the task
        starts, stops = _detect_sitting_tasks(acc_series, fs=100, peak_height=11, min_distance=10000)

        # cut the task
        tasks_sitting = _cut_segments(data, starts, stops, 3000)

        # store in tasks list
        tasks.extend(tasks_sitting)

    elif CABINETS in folder_name:

        # cut cabinets tasks and store in a list
        starts, stops = _detect_cabinets_tasks(acc_series, fs=100, peak_height=7, min_distance=20000)

        # cut the tasks
        tasks_cabinets = _cut_segments(data, starts, stops, 1000)

        # store in tasks list
        tasks.extend(tasks_cabinets)

    elif WALKING in folder_name:

        # cut walking tasks and store in a list
        envelope, starts, stops = _detect_walking_onset(acc_series, 100, 0.01)

        # validate the starts and stops
        valid_starts, valid_stops = _validate_walking_starts_stops(acc_series, folder_name, starts, stops, 12000, 30000)

        # cut segments
        tasks_walking = _cut_segments(data, valid_starts, valid_stops, 500)
        tasks.extend(tasks_walking)

    elif STANDING in folder_name:

        # cut standing tasks and store in a list.
        starts, stops = _detect_standing_tasks(acc_series, fs=100, peak_height=7, min_distance=10000)

        # cut the tasks
        tasks_standing = _cut_segments(data, starts, stops, 1000)

        # # first and last segments are the same task
        # # join the separated task into one df
        # task_standing_no_gestures = pd.concat([tasks_standing[0], tasks_standing[2]], ignore_index=True)

        tasks.extend(tasks_standing)

    elif STAIRS in folder_name:
        # cut walking tasks and store in a list
        envelope, starts, stops = _detect_walking_onset(acc_series, 100, 0.01)

        # validate the starts and stops
        valid_starts, valid_stops = _validate_walking_starts_stops(acc_series, folder_name, starts, stops, 2000, 10000)

        # cut segments
        tasks_stairs = _cut_segments(data, valid_starts, valid_stops, 250)

        # # join the separated tasks into one df
        # stairs_up = pd.concat([tasks_stairs[0], tasks_stairs[2]], ignore_index=True)
        # stairs_down = pd.concat([tasks_stairs[1], tasks_stairs[3]], ignore_index=True)

        tasks.extend(tasks_stairs)

    if not any(activity in folder_name for activity in SUPPORTED_ACTIVITIES):
        # If no supported activity is found, raise a ValueError
        raise ValueError(f"The activity: {folder_name} is not supported")

    return tasks


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _detect_walking_onset(y_acc, fs, threshold, envelope_type="rms", envelope_param=100):
    """
    gets the indices of where the walking tasks start and end based on the y-axis of the phone's accelerometer.
    :param y_acc: y-axis of the phone's accelerometer signal
    :param fs: sampling frequency
    :param threshold: the threshold used to detect the onset. Should be between [0, 1]. It is best to visualize the envelope
                      of the normalized signal in order to set this onset.
    :param envelope_type: the type of filter that should be used for getting the envelope of the signal. The following types
                          are available:
                          'lowpass': uses a lowpass filter
                          'ma': uses a moving average filter
                          'rms': uses a root-mean-square filter
    :param type_param: the parameter for the envelope_type. The following options are available (based on the envelope_type)
                       'lowpass': type_param is the cutoff frequency of the lowpass filter
                       'ma': type_param is the window size in samples
                       'rms': type_param is the window size in samples
    :return: the envelope, the start indices of the onsets, and the stop indices of the onsets
    """

    # apply butterworth filter
    y_acc = median_and_lowpass_filter(y_acc, fs)

    # # normalize the signal
    y_acc = y_acc / np.max(y_acc)

    # apply gravitational filter
    grav_component = gravitational_filter(y_acc, fs=100)

    # subtract gravitational component from acc
    y_acc = y_acc - grav_component

    # calculate the absolute value
    y_acc = np.abs(y_acc)

    # get the enevlope of the signal
    acc_env = get_envelope(y_acc, envelope_type=envelope_type, type_param=envelope_param)

    # binarize the signal
    binary_onset = (acc_env >= threshold).astype(int)

    # get the start and stopps of each task
    start_indices, stop_indices = _get_task_indices(acc_env, threshold)

    return acc_env, start_indices, stop_indices


def _get_task_indices(acc_env, threshold):
    """
    gets the indices for when the each walking task starts and stops. If the recording stops while walking is still performed,
    then there is no stop index for the last walking task.
    :param acc_env: the envelope of the accelerometer signal
    :param threshold: the threshold for defining what belongs to walking and what not. Any value of acc_env that is above the
                      the set threshold is considered as "walking" or "perfroming task". This value might have to be adapted
                      to each subject individually.
    :return: the start and stop indices of each performed task.
    """

    # binarize the signal
    binary_onset = (acc_env >= threshold).astype(int)

    # get the start and stopps of each task
    # (1) calculate the difference
    diff_sig = np.diff(binary_onset)

    # (2) get the task starts and end
    task_start = np.where(diff_sig == 1)[0]
    task_end = np.where(diff_sig == -1)[0]

    return task_start, task_end


def _detect_sitting_tasks(yacc, fs, peak_height, min_distance) -> Tuple[List[int], List[int]]:
    # butterworth filter
    yacc = median_and_lowpass_filter(yacc, fs)

    # get gravitational component
    yacc_grav = gravitational_filter(yacc, fs)

    # remove gravitational component
    yacc = yacc - yacc_grav

    # Find peaks
    peaks, _ = find_peaks(yacc, height=peak_height, distance=min_distance)

    # Ensure there's at least one peak for sitting
    if len(peaks) < 1:
        raise ValueError(
            "Less than one peak detected in the sitting aquisition. Adjust your peak detection parameters.")

    # Ensure there isn't more than two peaks for sitting
    if len(peaks) > 2:
        raise ValueError("Too many peaks were detected for the sitting activity. Adjust the peak detection parameters")

    # Define the cut points
    start_1 = peaks[0] + 1500
    end_1 = peaks[1] if len(peaks) > 1 else len(yacc)

    # Define the starts and stops
    starts = []
    stops = []

    starts.append(start_1)
    stops.append(end_1)

    return starts, stops


def _validate_walking_starts_stops(yacc, folder_name, starts, stops, min_value_between_starts_stops: int,
                                   min_stop_value: int):
    # validade the start indices to get only the starts of the tasks
    # Initialize an empty list to hold valid indices
    valid_starts = []

    # Iterate through the array, stopping at the second-to-last element
    for i in range(len(starts) - 1):
        # Check if the difference to the next element is less than 12000
        if starts[i + 1] - starts[i] >= min_value_between_starts_stops:
            valid_starts.append(starts[i])

    # append the last element since it has no next element to compare
    valid_starts.append(starts[-1])

    # Initialize an empty list to hold valid stop indices
    valid_stops = []

    # Iterate through the array, stopping at the second-to-last element
    for i in range(len(stops) - 1):
        # Check if the stop value is above min_stop_value and if the difference to the next element is greater than or equal to min_value_between_starts_stops
        if stops[i] >= min_stop_value and stops[i + 1] - stops[i] >= min_value_between_starts_stops:
            valid_stops.append(stops[i])

    # append the last value of stops indices
    valid_stops.append(stops[-1])

    # if the signal was cut early the last stop is the last value of the signal
    if len(valid_starts) != len(valid_stops):
        valid_stops.append(len(yacc))

    if WALKING in folder_name:
        # check for invalide starts and stops
        if len(valid_starts) > 3:
            raise ValueError(f"Too many starts for walking activity")

        elif len(valid_starts) < 3:
            raise ValueError(f"Not enough starts for walking activity")

        elif len(valid_stops) > 3:
            raise ValueError(f"Too many stops for walking activity")

        elif len(valid_stops) < 2:
            raise ValueError(f"Not enough stops for walking activity")

    return valid_starts, valid_stops


def _detect_cabinets_tasks(yacc, fs, peak_height, min_distance):
    # butterworth filter
    yacc = median_and_lowpass_filter(yacc, fs)

    # get gravitational component
    yacc_grav = gravitational_filter(yacc, fs)

    # remove gravitational component
    yacc = yacc - yacc_grav

    # Find peaks
    peaks, _ = find_peaks(yacc, height=peak_height, distance=min_distance)

    # Ensure there's at least two peaks for cabinets
    if len(peaks) < 2:
        raise ValueError(
            "Less than two peaks detected in the cabinets aquisition. Adjust your peak detection parameters.")

    # Ensure there isn't more than three peaks for cabinets
    if len(peaks) > 4:
        raise ValueError("Too many peaks were detected for the standing activity. Adjust the peak detection parameters")

    # Define the cut points
    start_1 = peaks[0] + 1500
    end_1 = peaks[1] - 500
    start_2 = peaks[1] + 500
    end_2 = (peaks[2] - 500) if len(peaks) > 2 else len(yacc)

    # Define the starts and stops
    starts = []
    stops = []

    starts.append(start_1)
    starts.append(start_2)
    stops.append(end_1)
    stops.append(end_2)

    return starts, stops


def _detect_standing_tasks(yacc, fs, peak_height, min_distance):
    # butterworth filter
    yacc = median_and_lowpass_filter(yacc, fs)

    # get gravitational component
    yacc_grav = gravitational_filter(yacc, fs)

    # remove gravitational component
    yacc = yacc - yacc_grav

    # Find peaks
    peaks, _ = find_peaks(yacc, height=peak_height, distance=min_distance)

    # Ensure we have at least three peaks for standing
    if len(peaks) < 3:
        raise ValueError(
            "Less than two peaks detected in the standing aquisition. Adjust the peak detection parameters.")

    # Ensure there isn't more than four peaks for standing
    elif len(peaks) > 4:
        raise ValueError("Too many peaks were detected for the standing activity. Adjust the peak detection parameters")

    # Define the cut points - for standing there's 4, can be 3 if a sensor stops early
    # if it's less then 3,this signal is trash
    first_start = peaks[0] + 1500
    first_stop = peaks[1] - 500
    second_start = peaks[1] + 500
    second_stop = peaks[2] - 500
    third_start = peaks[2] + 500
    third_stop = (peaks[3] - 500) if len(peaks) > 3 else len(yacc)

    # Define the starts and stops
    starts = []
    stops = []

    starts.append(first_start)
    starts.append(second_start)
    starts.append(third_start)

    stops.append(first_stop)
    stops.append(second_stop)
    stops.append(third_stop)

    return starts, stops


def _cut_segments(df: pd.DataFrame, starts: List[int], stops: List[int], trim_length: int):
    # list to store the tasks
    tasks = []

    for start, stop in zip(starts, stops):
        task = _cut_and_trim_task(df, start, stop, trim_length)
        tasks.append(task)

    return tasks


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


def _cut_and_trim_task(df: pd.DataFrame, start: int, end: int, trim_size: int) -> pd.DataFrame:
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

# def _cut_sitting_tasks(df: pd.DataFrame, recording_size: int = 90000, trim_length: int = 87000,
#                        end_samples: int = 500) -> pd.DataFrame:
#     """
#     Cuts the specified DataFrame to retain only the essential part of the sitting task recordings,
#     removes synchronization and connection-related data, and trims to the desired length. recording_size,
#     trim_size and end_samples should be chosen according to the duration of the acquisition.
#
#     Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. If sitting was done all
#     in one go, the trim_length should still be n * 87 000 samples to guarantee the same duration to all the
#     activities performed.
#
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data.
#     :param recording_size: int.
#         Duration in samples of the acquisition without connection/synchronization and stop at the end.
#     :param trim_length: int.
#         The length to which the signal should be trimmed.
#     :param end_samples:
#         Duration in samples of the stop before the end of the recording.
#     :return: pd.DataFrame.
#         Dataframe containing the sitting task only.
#     """
#     # cut end part
#     df = df.iloc[:-end_samples]
#     df.reset_index(drop=True, inplace=True)
#
#     # get only the actual recording, removes connection and sync parts
#     df = _cut_recording_time(df, recording_size)
#
#     # trim the data
#     df = _trim_dataframe(df, trim_length)
#
#     return df
#
#
# def _cut_cabinets_tasks(df: pd.DataFrame, recording_size: int = 91000, stop_size: int = 1000,
#                         end_samples: int = 1500) -> List[pd.DataFrame]:
#     """
#     Cuts the specified DataFrame to retain only the tasks performed during cabinets recordings,
#     removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
#     length. Segments the two cabinets tasks in this order: coffee and folders. recording_size, trim_size and
#     end_samples should be chosen according to the duration of the acquisition.
#
#     Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
#     have the same duration to guarantee that all activities are equally represented.
#
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data.
#     :param recording_size: int.
#         Duration in samples of the acquisition without connection/synchronization and stop at the end.
#     :param stop_size: int.
#         Duration in samples of the stops in between tasks.
#     :param end_samples:
#         Duration in samples of the stop before the end of the recording.
#     :return: List[pd.DataFrame].
#         List with two dataframes: the coffee task and folder task.
#     """
#     # Cut the end part
#     df = df.iloc[:-end_samples]
#     df.reset_index(drop=True, inplace=True)
#
#     # Get only the actual recording, removes connection and sync parts
#     df = _cut_recording_time(df, recording_size)
#
#     # Remove 10 seconds stop separating coffee and folder tasks
#     # First segment to keep
#     coffee_task_df = _cut_and_trim_task(df, 0, int(recording_size / 2 - stop_size/2), 1500)
#
#     # Second segment to keep
#     folder_task_df = _cut_and_trim_task(df, int(recording_size / 2 + stop_size/2), None, 1500)
#
#     # Store the DataFrames in a list and return
#     return [coffee_task_df, folder_task_df]
#
#
# def _cut_walking_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
#                        end_samples: int = 1500, segment_trim_size: int = 1000) -> List[pd.DataFrame]:
#     """
#     Cuts the specified DataFrame to retain only the tasks performed during walking recordings,
#     removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
#     length. Segments the three walking tasks/speeds in this order: slow, medium, and fast. recording_size, trim_size and
#     end_samples should be chosen according to the duration of the acquisition.
#
#     Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
#     have the same duration to guarantee that all activities are equally represented.
#
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data.
#     :param recording_size: int.
#         Duration in samples of the acquisition without connection/synchronization and stop at the end.
#     :param stop_size: int.
#         Duration in samples of the stops in between tasks.
#     :param end_samples:
#         Duration in samples of the stop before the end of the recording.
#     :return: List[pd.DataFrame].
#         List with three dataframes: slow, medium and fast walking speeds.
#     """
#     # Cut the end part
#     df = df.iloc[:-end_samples]
#     df.reset_index(drop=True, inplace=True)
#
#     # Get only the actual recording, removes connection and sync parts
#     df = _cut_recording_time(df, recording_size)
#
#     # Calculate recording size excluding stops
#     recording_size_no_stops = int(recording_size - 2 * stop_size)
#
#     # cut first walking task - slow walking speed
#     walk_slow_df = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 3), segment_trim_size)
#
#     # cut second walking task - medium walking speed
#     walk_medium_df = _cut_and_trim_task(df, int(recording_size_no_stops / 3 + stop_size),
#                                         int(2 * recording_size_no_stops / 3 + stop_size), segment_trim_size)
#
#     # cut third walking task - fast walking speed
#     walk_fast_df = _cut_and_trim_task(df, int(2 * recording_size_no_stops / 3 + 2 * stop_size),
#                                       None, segment_trim_size)  # Going to the end of the DataFrame
#
#     # Store the dataframes in a list
#     tasks_list = [walk_slow_df, walk_medium_df, walk_fast_df]
#
#     return tasks_list
#
#
# def _cut_standing_tasks(df: pd.DataFrame, recording_size: int = 92000, stop_size: int = 1000,
#                         end_samples: int = 1500, segment_trim_size: int = 1500) -> List[pd.DataFrame]:
#     """
#     Cuts the specified DataFrame to retain only the tasks performed during standing recordings,
#     removes synchronization and connection-related data, removes the stops in between tasks and trims to the desired
#     length. Segments the two standing tasks in this order: standing with gestures and standing without gestures.
#     recording_size, trim_size and end_samples should be chosen according to the duration of the acquisition.
#
#     Note: After trimming, there should be 14,5 minutes (87 000 samples) for each acquisition. All acquisitions should
#     have the same duration to guarantee that all activities are equally represented.
#
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data.
#     :param recording_size: int.
#         Duration in samples of the acquisition without connection/synchronization and stop at the end.
#     :param stop_size: int.
#         Duration in samples of the stops in between tasks.
#     :param end_samples:
#         Duration in samples of the stop before the end of the recording.
#     :return: List[pd.DataFrame].
#         List with two dataframes: standing with and without gestures.
#     """
#     # cut end part and get only the actual recording
#     df = df.iloc[:-end_samples]
#     df.reset_index(drop=True, inplace=True)
#     df = _cut_recording_time(df, recording_size)
#
#     # cut standing  where there were stops in the acquisition
#     recording_size_no_stops = int(recording_size - 2 * stop_size)
#
#     # cut first segment
#     standing_no_gestures_df1 = _cut_and_trim_task(df, 0, int(recording_size_no_stops / 4), int(segment_trim_size / 2))
#
#     # cut second segment
#     standing_with_gestures_df = _cut_and_trim_task(df, int(recording_size_no_stops / 4 + stop_size),
#                                                    int(3 * recording_size_no_stops / 4 + stop_size), segment_trim_size)
#
#     # cut third segment
#     standing_no_gestures_df2 = _cut_and_trim_task(df, int(3 * recording_size_no_stops / 4 + 2 * stop_size), None,
#                                                   int(segment_trim_size / 2))
#
#     # join the separated task into one df
#     standing_no_gestures_df = pd.concat([standing_no_gestures_df1, standing_no_gestures_df2], ignore_index=True)
#
#     # store tasks
#     tasks_list = [standing_with_gestures_df, standing_no_gestures_df]
#
#     return tasks_list

# def _reset_index_to_column(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Reset the index and convert it into a column called 'sec'.
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data with the index being the time in seconds.
#     :return: pd.DataFrame.
#         Dataframe with the previous index in seconds converted into a column 'sec' and with new index in samples.
#     """
#
#     # Move the current index to a column and reset
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'sec'}, inplace=True)  # Rename the new column to 'sec'
#     return df


# def _cut_recording_time(df: pd.DataFrame, recording_size: int) -> pd.DataFrame:
#     """
#     Cuts the data in the dataframe to keep the last samples that correspond to the recording time. Removes the initial
#     part pertaining to connection and synchronization.
#     :param df: pd.DataFrame.
#         Dataframe containing the sensor data to be cut
#     :param recording_size: int.
#         Duration in samples of the acquisition without connection/synchronization and stop at the end.
#     :return: pd.DataFrame.
#         Dataframe cut to the recording_size
#     """
#     df = df.tail(recording_size)
#     df.reset_index(drop=True, inplace=True)
#     return df
