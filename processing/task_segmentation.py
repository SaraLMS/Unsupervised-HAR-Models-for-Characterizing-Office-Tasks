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
    # cut synchronization jumps
    data = _cut_sync_jumps(data)

    # check type of activity
    if WALKING in folder_name:
        # cut pauses
        data = _cut_walking_tasks(data)

    elif STANDING in folder_name:
        data = _cut_standing_tasks(data)

    elif CABINETS in folder_name:
        data = _cut_cabinets_tasks(data)

    # cut the end pause
    data = _cut_end_part(data)

    if not any(activity in folder_name for activity in SUPPORTED_ACTIVITIES):
        # If no supported activity is found, raise a ValueError
        raise ValueError(f"The activity: {folder_name} is not supported")

    # reset time

    # new_time_axis = _generate_time_axis(data.iloc[:, 0].values)
    # data.index = new_time_axis

    return data


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _generate_time_axis(signal, sampling_rate=100):
    # get the number of samples
    num_samples = len(signal)

    # calculate the end of the signal in seconds
    end_time = num_samples / sampling_rate

    # generate the time axis
    time_axis = np.arange(0, end_time, 1 / sampling_rate)

    return time_axis

# # def _cut_time_ranges(df: pd.DataFrame, cut_ranges: List[Tuple[float, float]]):
# #     """ Removes specified time ranges from the DataFrame.
# #
# #     Args:
# #         df (pd.DataFrame): The DataFrame to cut from.
# #         cut_ranges (list of tuple): Each tuple contains start and end seconds to cut.
# #
# #     Returns:
# #         pd.DataFrame: The DataFrame after cutting the specified ranges.
# #     """
# #     mask = pd.Series(True, index=df.index)
# #     for start, end in cut_ranges:
# #         mask[(df.index >= start) & (df.index <= end)] = False
# #     return df[mask]


def _cut_time_ranges(df: pd.DataFrame, cut_ranges: List[Tuple[float, float]]) -> pd.DataFrame:
    """ Removes specified time ranges from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to cut from.
        cut_ranges (List[Tuple[int, int]]): Each tuple contains start and end seconds to cut.

    Returns:
        pd.DataFrame: The DataFrame after cutting the specified ranges.
    """
    # Sort ranges to ensure they are in the correct order and do not overlap
    cut_ranges.sort(key=lambda x: x[0])

    # Start with the full DataFrame and progressively remove the specified ranges
    current_start = df.index[0]
    result_df = pd.DataFrame()

    for start, end in cut_ranges:
        # Append the segment of df before the start of the cut
        if start > current_start:
            result_df = pd.concat([result_df, df.loc[current_start:start-1]])

        # Update the current_start to be the end of the current cut range
        current_start = end + 1

    # Append any remaining data after the last cut range
    if current_start <= df.index[-1]:
        result_df = pd.concat([result_df, df.loc[current_start:]])

    return result_df

def _cut_sync_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """ Cuts the initial synchronization jumps from the DataFrame. """
    return _cut_time_ranges(df, [(0.00, 35.00)])


def _cut_walking_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """ Cuts the pauses during walking tasks. """

    return _cut_time_ranges(df, [(350.00, 370.00), (665.00, 685.00)])


def _cut_standing_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """ Cuts the pauses during standing tasks. """
    return _cut_time_ranges(df, [(250.00, 270.00), (715.00, 735.00)])


def _cut_cabinets_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """ Cuts the pauses during cabinets tasks. """
    return _cut_time_ranges(df, [(495.00, 515.00)])


def _cut_end_part(df: pd.DataFrame) -> pd.DataFrame:
    """ Cuts the last 20 seconds from the DataFrame. """

    end_time = df.index[-1]
    return _cut_time_ranges(df, [(end_time - 20.00, end_time)])



