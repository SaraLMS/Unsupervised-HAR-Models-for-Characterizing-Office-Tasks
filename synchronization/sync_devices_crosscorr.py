# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from load.load_sync_data import load_used_devices_data
from parser.extract_from_path import get_folder_name_from_path
from .common import crop_dataframes_on_shift, join_dataframes_on_index, generate_filename, save_data_to_csv
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def sync_crosscorr(prefix: str, folder_path: str, output_path: str):
    """
    Synchronizes sensor data from two different devices based on cross correlation.
    Generates a new csv file containing all the synchronized sensor data from the two devices.

    Parameters:
        prefix (str):
        Prefix to add to the generated filename

        folder_path (str):
        Path to the folder containing the sensor data from the two devices.

        output_path (str):
        Path to the location where the file should be saved.
    """
    # get the dataframes of the signals in the folder
    dataframes_dic, datetimes_dic = load_used_devices_data(folder_path)

    # get the acc axis depending on the type of device
    acc_axis_array = _get_axis_from_acc(dataframes_dic)

    # get shift between the signals from different devices
    tau = _get_shift_to_synchronize_signals(acc_axis_array)

    # crop dataframes
    sync_signal_1, sync_signal_2 = crop_dataframes_on_shift(tau, dataframes_dic)

    # join signals into one dataframe
    df_joined = join_dataframes_on_index(sync_signal_1, sync_signal_2)

    # get folder name
    folder_name = get_folder_name_from_path(folder_path)

    # generate file name
    output_filename = generate_filename(datetimes_dic, folder_name, prefix, sync_type="crosscorr")

    # save csv file
    save_data_to_csv(output_filename, df_joined, output_path, folder_name)


def get_tau_crosscorr(folder_path: str) -> int:
    """
    Gets the shift in samples when synchronizing signals based on cross correlation.

    Parameters:
        folder_path (str):
        Path to the folder containing the sensor data from the two devices.

    Returns:
        Shift in samples calculated using cross correlation.
    """
    # get the dataframes of the signals in the folder
    dataframes_dic, datetimes_dic = load_used_devices_data(folder_path)

    # get the acc axis depending on the type of device
    acc_axis_array = _get_axis_from_acc(dataframes_dic)

    # get shift between the signals from different devices
    tau = _get_shift_to_synchronize_signals(acc_axis_array)
    return tau


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _get_axis_from_acc(dataframes_dic: Dict[str, pd.DataFrame], window_range: Tuple[int, int] = (0, 10000)) -> List[
    pd.Series]:
    """
    Gets the accelerometer axis from the devices used for synchronization

    Parameters:
        dataframes_dic (Dict[str, pd.DataFrame]):
        Dictionary containing the chosen device names as keys and sensor data from said devices as values.

        window_range (Tuple[int,int]):
        Window of samples of the accelerometer axis data to be used for synchronization.

    Returns:
        List containing the axis for synchronization
    """
    # get the start and end values of the axis - window of samples containing the jumps for cross corr
    start, end = window_range
    # TODO ASK PHILLIP ABOUT THIS - no magic numbers
    acc_axis_array = []

    for device, df in dataframes_dic.items():

        if device == 'phone':

            # column = 'yAcc'
            # factor = 1

            axis_to_sync = df['yAcc'][start:end]
            acc_axis_array.append(axis_to_sync)

        elif device == 'watch':

            axis_to_sync = -1 * df['xAcc_wear'][start:end]
            acc_axis_array.append(axis_to_sync)

        elif device == 'mban':
            pass
    # return factor, column
    return acc_axis_array


def _get_shift_to_synchronize_signals(acc_axis_array: List[pd.Series]) -> int:
    """
        -----
        Brief
        -----
        This function synchronises the input signals using the full cross correlation function between the signals.

        -----------
        Description
        -----------
        Signals acquired with two devices may be dephased. It is possible to synchronise the two signals by multiple
        methods. Here, it is implemented a method that uses the calculus of the cross-correlation between those signals and
        identifies the correct instant of synchrony.

        This function synchronises the two input signals and returns the dephasing between them, and the resulting
        synchronised signals.

        ----------
        Parameters
        ----------
        in_signal_1 : list or numpy.array
            One of the input signals.
        in_signal_2 : list or numpy.array
            The other input signal.

        Returns
        -------
        phase : int
            The dephasing between signals in data points.
        result_signal_1: list or numpy.array
            The first signal synchronised.
        result_signal_2: list or numpy.array
            The second signal synchronised.
        """
    # get the pd.Series
    # order is irrelevant
    in_signal_1 = acc_axis_array[0]
    in_signal_2 = acc_axis_array[1]
    # signal normalisation
    mean_1, std_1, mean_2, std_2 = [np.mean(in_signal_1), np.std(in_signal_1), np.mean(in_signal_2),
                                    np.std(in_signal_2)]
    signal_1 = in_signal_1 - mean_1
    signal_1 /= std_1
    signal_2 = in_signal_2 - mean_2
    signal_2 /= std_2

    # zero padding signals so that they are of same length, this facilitates the calculation because
    # then the delay between both signals can be directly calculated
    # zero padding only if needed

    if (len(signal_1) != len(signal_2)):

        # check which signal has to be zero padded
        if (len(signal_1) < len(signal_2)):

            # pad first signal
            signal_1 = np.append(signal_1, np.zeros(len(signal_2) - len(signal_1)))

        else:

            # pad second signal
            signal_2 = np.append(signal_2, np.zeros(len(signal_1) - len(signal_2)))

    # Calculate the full cross-correlation between the two signals.
    correlation = np.correlate(signal_1, signal_2, 'full')

    # crop signals to original length (removing zero padding)
    signal_1 = signal_1[:len(in_signal_1)]
    signal_2 = signal_2[:len(in_signal_2)]

    # calculate tau / shift between both signals
    tau = int(np.argmax(correlation) - (len(correlation)) / 2)

    return tau
