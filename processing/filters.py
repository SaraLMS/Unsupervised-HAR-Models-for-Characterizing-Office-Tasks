# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import scipy as scp
from scipy import signal


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def median_and_lowpass_filter(sensor_data: np.ndarray, fs: int, medfilt_window_length=11) -> np.ndarray:
    """
    First a median filter is applied and then a
    filter with a cutoff frequency of 20 Hz is applied.
    The processing scheme is based on:
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


def get_envelope(emg_series, envelope_type='lowpass', type_param=10):
    """
    Gets the envelope of the passed EMG signal. There are three types available
    1. 'lowpass': uses a lowpass filter
    2. 'ma': uses a moving average filter
    3. 'rms': uses a root-mean-square filter
    :param emg_series: the EMG data
    :param envelope_type: the type of filter that should be used for getting the envelope as defined above
    :param type_param: the parameter for the envelope_type. The following options are available (based on the envelope_type)
                       'lowpass': type_param is the cutoff frequency of the lowpass filter
                       'ma': type_param is the window size in samples
                       'rms': type_param is the window size in samples
    :return: pandas series containing the envelope of the EMG
    """

    # check for the passed type
    if envelope_type == 'lowpass':
        # apply lowpass filter
        emg_series = _butter_lowpass_filter(emg_series, cutoff=10, fs=100)
    elif envelope_type == 'ma':
        # apply moving average
        emg_series = _moving_average(emg_series, wind_size=type_param)
    elif envelope_type == 'rms':
        # apply rms
        emg_series = _window_rms(emg_series, window_size=type_param)

    else:
        # undefined filter type passed
        IOError('the type you chose is not defined.')

    return pd.Series(emg_series)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Filters a signal using a butterworth lowpass filter
    :param data: the data that should be filtered
    :param cutoff: frequency cutoff
    :param fs: sampling frequency
    :param order: order of the filter
    :return:
    """
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def _moving_average(data, wind_size=3):
    """
    -----
    Brief
    -----
    Application of a moving average filter for signal smoothing.

    -----------
    Description
    -----------
    In certain situations it will be interesting to simplify a signal, particularly in cases where
    some events with a random nature take place (the random nature of EMG activation periods is
    a good example).

    One possible simplification procedure consists in smoothing the signal in order to obtain
    only an "envelope". With this methodology the analysis is mainly centered on seeing patterns
    in data and excluding noise or rapid events [1].

    The simplification can be achieved by segmenting the time series in multiple windows and
    from each window an average value of all the samples that it contains will be determined
    (dividing the sum of all sample values by the window size).

    A quick and efficient implementation (chosen in biosignalsnotebooks package) of the moving window
    methodology is through a cumulative sum array.

    [1] https://en.wikipedia.org/wiki/Smoothing

    ---------
    Parameters
    ----------
    data : list
        List of signal samples.
    wind_size : int
        Number of samples inside the moving average window (a bigger value implies a smoother
        output signal).

    Returns
    -------
    out : numpy array
        Array that contains the samples of the smoothed signal.
    """

    wind_size = int(wind_size)
    ret = np.cumsum(data, dtype=float)
    ret[wind_size:] = ret[wind_size:] - ret[:-wind_size]
    return np.concatenate((np.zeros(wind_size - 1), ret[wind_size - 1:] / wind_size))


def _window_rms(data, window_size=3):
    """
    Passes a root-mean-square filter over the data.
    :param data: the data for which the root-mean-square should be calculated
    :param window_size: the window size
    :return: the rms for the given window
    """
    data_squared = np.power(data, 2)
    window = np.ones(window_size) / float(window_size)
    return np.sqrt(np.convolve(data_squared, window, 'valid'))


def _butter_lowpass(cutoff, fs, order=4):
    """
    Creates a butterworth lowpass filter
    :param cutoff: frequency cutoff
    :param fs: sampling rate
    :param order: order of the filter
    :return: butterworth lowpass filter
    """
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
