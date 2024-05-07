# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
from scipy.signal import butter, medfilt, sosfilt
import scipy as scp

from constants import ACCELEROMETER_PREFIX, SUPPORTED_PREFIXES


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def apply_filters(data: pd.DataFrame, fs: int) -> pd.DataFrame:
    """
    Applies various filters to sensor data columns in a CSV file.

    This function processes each sensor data column in the file, applying median and lowpass filters.
    For accelerometer data, it additionally removes the gravitational component.

    Parameters:
        data (pd.DataFrame): DataFrame containing the sensor data.
        fs (int): The sampling frequency of the sensor data.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered sensor data, with the same structure as the input file.
    """

    filtered_data = data.copy()

    # Process each sensor column directly
    for sensor in filtered_data.columns:

        # Determine if the sensor is an accelerometer or gyroscope by its prefix
        if any(prefix in sensor for prefix in SUPPORTED_PREFIXES):
            # Get raw sensor data
            raw_data = filtered_data[sensor].values

            # Apply median and lowpass filters
            filtered_median_lowpass_data = _median_and_lowpass_filter(raw_data, fs)

            if ACCELEROMETER_PREFIX in sensor:
                # For accelerometer data, additionally remove the gravitational component
                gravitational_component = _gravitational_filter(raw_data, fs)

                # Remove gravitational component from filtered data
                filtered_median_lowpass_data -= gravitational_component

            # Update DataFrame with filtered sensor data
            filtered_data[sensor] = pd.Series(filtered_median_lowpass_data, index=filtered_data.index)

    # remove first 200 samples to remove impulse response of the bandpass filter
    filtered_data = filtered_data.iloc[200:]

    return filtered_data


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _median_and_lowpass_filter(sensor_data: np.ndarray, fs: int, medfilt_window_length=11) -> np.ndarray:
    """
    First a median filter is applied and then a 3rd order butterworth lowpass
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
    filt = butter(order, f_c, fs=fs, output='sos')

    # copy the array
    filtered_data = sensor_data.copy()

    # check the dimensionality of the input
    if filtered_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(filtered_data.shape[1]):
            # get the channel
            sig = sensor_data[:, channel]

            # apply the median filter
            sig = medfilt(sig, medfilt_window_length)

            # apply butterworth filter
            filtered_data[:, channel] = sosfilt(filt, sig)

    else:  # 1-D array

        # apply median filter
        med_filt = medfilt(sensor_data, medfilt_window_length)

        # apply butterworth filter
        filtered_data = sosfilt(filt, med_filt)

    return filtered_data


def _gravitational_filter(acc_data: np.ndarray, fs: int) -> np.ndarray:
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
    filter = butter(order, f_c, fs=fs, output='sos')

    # copy the array
    gravity_data = acc_data.copy()

    # check the dimensionality of the input
    if gravity_data.ndim > 1:  # (MxN) array

        # cycle of the channels contained in data
        for channel in range(gravity_data.shape[1]):
            # get the channel
            sig = acc_data[:, channel]

            # apply butterworth filter
            gravity_data[:, channel] = sosfilt(filter, sig)

    else:  # 1-D array

        gravity_data = sosfilt(filter, acc_data)

    return gravity_data
