# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from load.load_sync_data import load_used_devices_data
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _get_signals_dataframes(folder_path: str) -> Dict[str, pd.DataFrame]:
    dataframes_dic = load_used_devices_data(folder_path)

    return dataframes_dic


def _get_axis_from_acc(dataframes_dic: Dict[str, pd.DataFrame]) -> List[pd.Series]:


    acc_axis_array = []

    for device, df in dataframes_dic.items():

        if device == 'phone':

            axis_to_sync = df['yAcc'][0:500] ########### ASK PHILLIP ABOUT THIS!!!!!!!!!!!!!!!!!!!!!!!!!!
            acc_axis_array.append(axis_to_sync)

        elif device == 'watch':

            axis_to_sync = -1 * df['xAcc_wear'][0:500]
            acc_axis_array.append(axis_to_sync)

        elif device == 'mban':
            pass

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
