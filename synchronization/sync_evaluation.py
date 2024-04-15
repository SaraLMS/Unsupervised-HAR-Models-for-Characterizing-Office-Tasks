# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from synchronization.sync_devices_crosscorr import get_tau_crosscorr
from synchronization.sync_devices_timestamps import get_tau_filename, get_tau_logger, check_logger_timestamps
from synchronization.sync_parser import check_logger_file


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

# def sync_evaluation(logger_folder_path: str, sync_folder_path: str, selected_sensors: Dict[str, List[str]]) -> pd.DataFrame:
#     """
#     Evaluates the performance of the three synchronization methods: cross correlation, filename timestamps
#     and logger file timestamps.
#
#     Parameters:
#         logger_folder_path: str
#         Path to the folder containing the logger file
#
#         sync_folder_path: str
#         Path to the folder containing the synchronized android sensors data
#
#         selected_sensors: Dict[str, List[str]]
#         Dictionary containing the sensors and devices chosen
#
#     Returns:
#         report (pd.DataFrame): a pandas DataFrame with the following [column]: row
#                          [Cross correlation shift]: The shift in samples calculated using the cross correlation method.
#
#                          [Filename timestamps shift]: The shift in samples calculated using the filenames start times.
#
#                          [logger timestamps shift]: The shift in samples calculated using the logger file start times.
#
#                          [mean]: The mean of the shift between signals.
#
#                          [Distance between crosscor and filename shift]: |crosscorr shift - filenames shift|.
#
#                          [Distance between crosscor and logger shift]: |crosscorr shift - logger shift|.
#
#                          [Best timestamps method]: min(distance filename shift, distance logger shift).
#
#                          [Filenames position]: if filenames method precedes or lags the cross correlation.
#
#                          [Logger position]: if logger method precedes or lags the cross correlation.
#
#     """
#     # calculate shift with cross correlation method
#     crosscorr_tau = get_tau_crosscorr(sync_folder_path)
#
#     # calculate shift with filename start times method
#     filename_tau = get_tau_filename(sync_folder_path)
#
#     # calculate shift with logger file start times method
#     logger_tau = get_tau_logger(logger_folder_path, sync_folder_path)
#
#     # calculate the mean of the shift between signals
#     shift_array = [crosscorr_tau, filename_tau, logger_tau]
#     mean = _calc_mean(shift_array)
#
#     # calculate the difference in seconds to the cross correlation method
#     filename_diff = np.abs(crosscorr_tau - filename_tau)
#     logger_diff = np.abs(crosscorr_tau - logger_tau)
#
#     # check if filename method lags or precedes cross correlation
#     if np.sign(filename_tau-crosscorr_tau) == -1:
#         filename_position = "precedes"
#     else:
#         filename_position = "lags"
#
#     # check if logger method lags or precedes cross correlation
#     if np.sign(logger_tau - crosscorr_tau) == -1:
#         logger_position = "precedes"
#     else:
#         logger_position = "lags"
#
#     # calculate the min distance to the cross correlation method
#     min_distance = min(filename_diff, logger_diff)
#     if min_distance == filename_diff:
#         best = 'Filename timestamps'
#     else:
#         best = 'Logger timestamps'
#
#     # create dictionary
#     report = {
#         'Cross correlation shift': crosscorr_tau,
#         'Filename timestamps shift': filename_tau,
#         'Logger timestamps shift': logger_tau,
#         'Shift mean': mean,
#         'Distance between cross correlation and filename shift': filename_diff,
#         'Distance between cross correlation and logger shift': logger_diff,
#         'Filenames position in relation to cross correlation': filename_position,
#         'Logger position in relation to cross correlation': logger_position,
#         'Best performing method after cross correlation': best,
#     }
#     # Turn the dictionary into a DataFrame
#     report_df = pd.DataFrame([report])
#
#     return report_df
def sync_evaluation(logger_folder_path: str, sync_folder_path: str, selected_sensors: list) -> pd.DataFrame:
    """
    Evaluates the performance of three synchronization methods: cross correlation, filename timestamps,
    and logger file timestamps.

    Parameters:
        logger_folder_path: str
            Path to the folder containing the logger file

        sync_folder_path: str
            Path to the folder containing the synchronized android sensors data

        selected_sensors (Dict[str, List[str]):
            Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    Returns:
        report (pd.DataFrame): a pandas DataFrame with detailed synchronization performance metrics.
    """
    # calculate shift with cross correlation method
    crosscorr_tau = get_tau_crosscorr(sync_folder_path)

    # calculate shift with filename start times method
    filename_tau = get_tau_filename(sync_folder_path)

    # check if the logger file is available, not empty, and has all required timestamps
    if check_logger_file(logger_folder_path) and check_logger_timestamps(logger_folder_path, sync_folder_path,
                                                                         selected_sensors):
        logger_tau = get_tau_logger(logger_folder_path, sync_folder_path)
        logger_diff = np.abs(crosscorr_tau - logger_tau)

        if np.sign(logger_tau - crosscorr_tau) == -1:
            logger_position = "precedes"
        else:
            logger_position = "lags"
    else:
        logger_tau = None
        logger_diff = None
        logger_position = "N/A"

    # calculate the mean of the shift between signals, ignoring None values
    shifts = [s for s in [crosscorr_tau, filename_tau, logger_tau] if s is not None]
    mean = np.mean(shifts) if shifts else None

    # calculate the difference in seconds to the cross correlation method
    filename_diff = np.abs(crosscorr_tau - filename_tau)

    # check if filename method lags or precedes cross correlation
    if np.sign(filename_tau - crosscorr_tau) == -1:
        filename_position = "precedes"
    else:
        filename_position = "lags"

    # determine the best performing method after cross correlation
    if logger_diff is not None:
        min_distance = min(filename_diff, logger_diff)
        best = 'Filename timestamps' if min_distance == filename_diff else 'Logger timestamps'
    else:
        min_distance = filename_diff
        best = 'Filename timestamps'

    # create dictionary
    report = {
        'Cross correlation shift': crosscorr_tau,
        'Filename timestamps shift': filename_tau,
        'Logger timestamps shift': logger_tau,
        'Shift mean': mean,
        'Distance between cross correlation and filename shift': filename_diff,
        'Distance between cross correlation and logger shift': logger_diff,
        'Filenames position in relation to cross correlation': filename_position,
        'Logger position in relation to cross correlation': logger_position,
        'Best performing method after cross correlation': best,
    }
    # Turn the dictionary into a DataFrame
    report_df = pd.DataFrame([report])

    return report_df


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _calc_mean(values_array: List[int]) -> int:
    """
    Calculates and returns the mean (average) of a list of integer values, rounded to the nearest integer.

    Parameters:
    values_array (List[int]): A list of integer values for which the mean is to be calculated.

    Returns:
    int: The mean of the input values, rounded to the nearest integer.
    """
    if not values_array:
        raise ValueError("The input array is empty")
    return round(sum(values_array) / len(values_array))
