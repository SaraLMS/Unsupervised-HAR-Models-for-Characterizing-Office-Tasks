


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd
import numpy as np
import warnings
import json
from typing import List, Tuple, Dict, Any


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_device_data(in_path: List[str], print_report: bool = False) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Function to load data from a single or multiple android files
    Parameters
    ----------
    in_path (list of strings or string): string or list containing the path(s) to the files that are supposed to be loaded
    print_report (boolean): boolean indicating to print the report that is generated while loading the data

    Returns
    -------
    sensor_data (list): a list containing the sensor data (including the time axis). The list has the same length as
                        the amount of sensors loaded.

    report (dictionary): a dictionary with the following fields
                         [names]: The names of the sensors in the order they were loaded into the list.

                         [column names]: The names of the columns in the order they were loaded into the list.

                         [number of samples]: The number of samples each sensor recorded.

                         [starting times]: The timestamps when the sensors started recording.

                         [stopping times]: The timestamps when the sensors stopped recording.

                         [avg. sampling rates]: The average sampling rate of each sensor (*).

                         [min. sampling rate]: The minimum sampling rate (of all sensors).

                         [max. sampling rate]: The maximum sampling rate (of all sensors).

                         [mean sampling rate]: The mean of the sampling rates.

                         [std. sampling rate]: The standard deviation of the sampling rates.

                         [starting order]: Order in which the sensors started recording, from first to last.

                         [stopping order]: Order in which the sensors stopped recording, from first to last.
    """

    # boolean for checking if a single file was loaded
    single_file = False

    # check if in_path is a string
    if isinstance(in_path, str):
        # set boolean for indicating that it was a single file
        single_file = True

        # put the in_path string into a list (in order to cycle over it later)
        in_path = [in_path]

    # list for holding the data of all sensors
    sensor_data = []

    # list for holding sensor names
    names = []

    # list for holding column names
    column_names = []

    # list for holding the number of samples each sensor recorded
    num_samples = []

    # list for holding average sampling rates
    avg_sampling_rates = []

    # list for holding start times
    start_times = []

    # list for holding stop times
    stop_times = []

    # cycle over the files
    for file in in_path:

        # suppress loadtxt warning that is thrown when there is no sensor data present in the file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # get columns indices from header
            if "ACCELEROMETER" in file or "GYROSCOPE" in file or "MAGNETIC_FIELD" in file or "MAGNETOMETER" in file:
                column_indices = [0, 1, 2, 3]

            elif "ROTATION_VECTOR" in file:
                column_indices = [0, 1, 2, 3, 4]

            elif "HEART_RATE" in file or "NOISERECORDER" in file:
                column_indices = [0, 1]

            data = pd.read_csv(file, delimiter="\t", header=None, skiprows=3)[column_indices]
            data = data.drop_duplicates(subset=[0])
            data = data.to_numpy()

        # check if data array has values
        # (the signifcant motion sensor might not return any data if no significant motion was detected)
        # the sensor is only loaded and added to the report if it has at least one sampled data point
        if (data.size):

            # get the sensor data
            sensor_data.append(data)

            # get the dimensionality of the data
            if (data.ndim == 1):  # 1D array, this means that the sensor only sampled a single point

                # get the time axis
                time_axis = data[:1]

                # set the sampling rate to zero because only one sample was acquired by the sensor
                avg_sampling_rates.append(0)

            else:  # multi-dimensional

                # get the times axis
                time_axis = data[:, 0]

                # calculate the average sampling rate (the time axis in the files is in nanoseconds)
                # the sampling rate will not be rounded in order to show the 'true' average sampling rate
                avg_sampling_rates.append(calc_avg_sampling_rate(time_axis, unit='nanoseconds', round=False))

            # get the number of samples
            num_samples.append(time_axis.size)

            # get the start time of the signal
            start_times.append(time_axis[0])

            # get the stop time of the signal
            stop_times.append(time_axis[-1])

            # open the file and retrieve information from the header
            with open(file, encoding='latin-1') as opened_file:
                # read the information from the header lines (omitting the begin and end tags of the header)
                header_string = opened_file.readlines()[1][2:]  # omit "# " at the beginning of the sensor information

                # convert the header into a dict
                header = json.loads(header_string)

                # get the device name by reading the first key
                device_name = next(iter(header))

                # add the name of the sensor to the list
                name = header[device_name]['sensor'][0]

                # remove the x from the name (in case it is a 3-axis sensor)
                if (name.startswith('x')): name = name[1:]

                # add the name to the list
                names.append(name)

                # get all sensor column names
                columns = header[device_name]['column']

                # add column to the list
                column_names.append(columns)

    # calculate max, min, mean and std
    max_sample = np.max(avg_sampling_rates)
    min_sample = np.min(avg_sampling_rates)
    mean = np.mean(avg_sampling_rates)
    std = np.std(avg_sampling_rates)

    # get the starting order of the sensors
    starting_order = [name for (_, name) in sorted(zip(start_times, names))]

    # get the stopping order of the sensors
    stopping_order = [name for (_, name) in sorted(zip(stop_times, names))]

    # create dictionary
    report = {
        'names': names,
        'column names': column_names,
        'number of samples': num_samples,
        'starting times': start_times,
        'stopping times': stop_times,
        'avg. sampling rates': avg_sampling_rates,
        'min. sampling rate': min_sample,
        'max. sampling rate': max_sample,
        'mean sampling rate': mean,
        'std. sampling rate': std,
        'starting order': starting_order,
        'stopping order': stopping_order,
    }

    # print a report if the user indicates to do so
    if (print_report): [print('{}: {}'.format(key, value)) for key, value in report.items()]

    # check if single file was loaded in that case return sensor_data as an array instead of a list
    if single_file:
        sensor_data = sensor_data[0]

    return sensor_data, report


def calc_avg_sampling_rate(time_axis: np.ndarray, unit: str = 'seconds', round: bool = True) -> float:
    """
    function to calculate the average sampling rate of signals recorded with an android sensor. The sampling rate is
    rounded to the next tens digit if specified(i.e 34.4 Hz = 30 Hz | 87.3 Hz = 90 Hz).
    sampling rates below 5 Hz are set to 1 Hz.

    Parameters
    ----------
    time_axis (N array_like): The time axis of the sensor
    unit (string, optional): the unit of the time_axis. Either 'seconds' or 'nanoseconds' can be used.
                             If not specified 'seconds' is used
    round (boolean, true): Boolean to indicate whether the sampling rate should be rounded to the next tens digit

    Returns
    -------
    avg_sampling_rate: the average sampling rate of the sensor

    """

    # check the input for unit and set the dividend accordingly
    if(unit == 'seconds'):

        dividend = 1

    elif(unit == 'nanoseconds'):

        dividend = 1e9

    else:  # invalid input

        raise IOError('The value for unit is not valid. Use either seconds or nanoseconds')

    # calculate the distance between sampling points
    # data[:,0] is the time axis
    sample_dist = np.diff(time_axis)

    # calculate the mean distance
    mean_dist = np.mean(sample_dist)

    # calculate the sampling rate and add it to the list
    # 1e9 is used because the time axis is in nanoseconds
    avg_sampling_rate = dividend / mean_dist

    # round the sampling rate if specified
    if(round):
        avg_sampling_rate = round_sampling_rate(avg_sampling_rate)

    return avg_sampling_rate


def round_sampling_rate(sampling_rate: float) -> int:
    """
    Function for round the sampling rate to the nearest tens digit. Sampling rates below 5 Hz are set to 1 Hz

    Parameters
    ----------
    sampling_rate: A sampling rate

    Returns
    -------
    rounded_sampling rate: the sampling rounded to the next tens digit
    """
    # check if sampling rate is below 5 Hz in that case always round to one
    if sampling_rate < 5:

        # set sampling rate to 1
        rounded_sampling_rate = 1

    else:

        # round to the nearest 10 digit
        rounded_sampling_rate = round(sampling_rate/10) * 10

    return rounded_sampling_rate
