# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
from scipy.signal import butter, medfilt, sosfilt
import scipy as scp


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def median_and_lowpass_filter(sensor_data: np.ndarray, fs: int, medfilt_window_length=11) -> np.ndarray:
    """
    First a median filter is applied and then a 3rd order butterworth lowpass
    filter with a cutoff frequency of 20 Hz is applied.
    The filtering scheme is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
    :param sensor_data: a 1-D or (MxN) array, where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :param medfilt_window_length: the length of the median filter. Has to be odd.
    :return: the filtered data
    """

    # define the filter
    order = 3
    f_c = 20
    filt = scp.signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    filtered_data = sensor_data.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = sensor_data[:, channel]

            # apply the median filter
            sig = scp.signal.medfilt(sig, medfilt_window_length)

            # apply butterworth filter
            filtered_data[:, channel] = scp.signal.sosfilt(filt, sig)

    else:  # 1-D array

        # apply median filter
        med_filt = scp.signal.medfilt(sensor_data, medfilt_window_length)

        # apply butterworth filter
        filtered_data = scp.signal.sosfilt(filt, med_filt)

    return filtered_data


def gravitational_filter(acc_data: np.ndarray, fs: int) -> np.ndarray:
    """
    Function to filter out the gravitational component of ACC signals using a 3rd order butterworth lowpass filter with
    a cuttoff frequency of 0.3 Hz
    The implementation is based on:
    "A Public Domain Dataset for Human Activity Recognition Using Smartphones"
    https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
    :param acc_data: a 1-D or (MxN) array, where where M is the signal length in samples and
                 N is the number of signals / channels.
    :param fs: the sampling frequency of the acc data.
    :return: the gravitational component of each signal/channel contained in acc_data
    """

    # define the filter
    order = 3
    f_c = 0.3
    filter = scp.signal.butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = scp.signal.sosfilt(filter, sig)

    else:  # 1-D array

        gravity_data = scp.signal.sosfilt(filter, acc_data)

    return gravity_data
