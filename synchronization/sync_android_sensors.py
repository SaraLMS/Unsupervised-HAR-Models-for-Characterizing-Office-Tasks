# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from load.load_raw_data import load_device_data, calc_avg_sampling_rate, round_sampling_rate
from .common import create_dir, get_folder_name_from_path
import numpy as np
import pandas as pd
import os
import scipy.interpolate as scp
from typing import List, Dict, Tuple


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def sync_all_classes(data_path: str, out_path: str, selected_sensors: Dict[str, List[str]]) -> None:
    """
    Sync sensor data for all classes of movements on a single device.

    Parameters
    ----------
    data_path : str
        The path to the main folder containing the raw sensor data.

    out_path : str
        The path to the main folder where synchronized data will be saved.

    selected_sensors : Dict[str, List[str]]
        A dictionary where keys are device modalities ("phone", "watch") and values are lists of sensor
        names to be synchronized. Supported sensor names include:
            - "acc": accelerometer
            - "gyr": gyroscope
            - "mag": magnetometer
            - "rotvec": rotation vector
            - "wearheartrate": heart rate (applicable only for the "watch" device)
            - "noise": ambient noise (applicable only for the "phone" device)

    Returns
    -------
    None - This function generate csv files with the synchronized data
    """
    # iterate through the folders in the main path
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        _sync_all_sensors_in_class(folder_path, out_path, selected_sensors)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _get_sensor_names(device: str, sensor: str) -> str:
    """
    Generate sensor names that fit the opensignals file name convention.

    Parameters
    ----------
    device: str
               The modality of the device used for the acquisitions The following sensor names are supported:
               "phone": smartphone
               "watch": smartwatch

    sensor: str
                 The sensor which should be loaded from the data. The following sensor names are supported:
                 "acc": accelerometer
                 "gyr": gyroscope
                 "mag": magnetometer
                 "rotvec": rotation vector
                 "wearheartrate": heart rate (smartwatch only)
                 "noise": ambient noise (smartphone only)

    Returns
    -------
    Sensor name in opensignals format.
    """

    if device == "phone":

        device_name = "ANDROID"

        if sensor == "acc":
            sensor_name = "ACCELEROMETER"

        elif sensor == "gyr":
            sensor_name = "GYROSCOPE"

        elif sensor == "mag":
            sensor_name = "MAGNETIC_FIELD"

        elif sensor == "rotvec":
            sensor_name = "ROTATION_VECTOR"

        elif sensor == "noise":
            sensor_name = "NOISERECORDER"

        else:
            raise ValueError(f"Invalid sensor: {sensor}")

    elif device == "watch":

        device_name = "ANDROID_WEAR"

        if sensor == "acc":
            sensor_name = "ACCELEROMETER"

        elif sensor == "gyr":
            sensor_name = "GYROSCOPE"

        elif sensor == "wearheartrate":
            sensor_name = "HEART_RATE"

        elif sensor == "mag":
            sensor_name = "MAGNETOMETER"

        elif sensor == "rotvec":
            sensor_name = "ROTATION_VECTOR"

        else:
            raise ValueError(f"Invalid sensor: {sensor}")

    one_sensor_name = f"opensignals_{device_name}_{sensor_name}"

    # device == "muscleban"
    # implement
    return one_sensor_name


def _re_sample_data(time_axis, data, start=0, stop=-1, shift_time_axis=True, sampling_rate=100,
                    kind_interp='quadratic'):
    """
    function to re-sample android sensor data from a non-equidistant sampling to an equidistant sampling
    Parameters
    ----------
    time_axis (N, array_like): A 1D array containing the original time axis of the data

    data (...,N,..., array_like): A N-D array containing data columns that are supposed to be interpolated.
                                  The length of data along the interpolation axis has to be the same size as time.

    start (int, optional): The sample from which the interpolation should be started. When not specified the
                           interpolation starts at 0. When specified the signal will be cropped to this value.

    stop (int, optional): The sample at which the interpolation should be stopped. When not specified the interpolation
                          stops at the last value. When specified the signal will be cropped to this value.

    shift_time_axis (bool, optional): If true the time axis will be shifted to start at zero and will be converted to seconds.

    sampling_rate (int, optional): The sampling rate in Hz to which the signal should be re-sampled.
                                   The value should be > 0.
                                   If not specified the signal will be re-sampled to the next tens digit with respect to
                                   the approximate sampling rate of the signal (i.e. approx. sampling of 99.59 Hz will
                                   be re-sampled to 100 Hz).

    kind_interp (string, optional): Specifies the kind of interpolation method to be used as string.
                                    If not specified, 'linear' interpolation will be used.
                                    Available options are: ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                                    ‘previous’, ‘next’.

    Returns
    -------

    the new time_axis, the interpolated data, and the sampling rate

    """

    # crop the data and time to specified start and stop values
    if start != 0 or stop != -1:
        time_axis = time_axis[start:stop]

        # check for dimensionality of the data
        if data.ndim == 1:  # 1D array

            data = data[start:stop]

        else:  # multidimensional array

            data = data[start:stop, :]

    # get the original time origin
    time_origin = time_axis[0]

    # shift time axis (shifting is done in order to simplify the calculations)
    time_axis = time_axis - time_origin
    time_axis = time_axis * 1e-9

    # calculate the approximate sampling rate and round it to the next tens digit
    if sampling_rate is None:
        # get the average sampling rate
        sampling_rate = calc_avg_sampling_rate(time_axis)

    # create new time axis
    time_inter = np.arange(time_axis[0], time_axis[-1], 1 / sampling_rate)

    # check for the dimensionality of the data array.
    if data.ndim == 1:  # 1D array

        # create the interpolation function
        inter_func = scp.interp1d(time_axis, data, kind=kind_interp)

        # calculate the interpolated column and save it to the correct column of the data_inter array
        data_inter = inter_func(time_inter)

    else:  # multidimensional array

        # create dummy array
        data_inter = np.zeros([time_inter.shape[0], data.shape[1]])

        # cycle over the columns of data
        for col in range(data.shape[1]):
            # create the interpolation function
            inter_func = scp.interp1d(time_axis, data[:, col], kind=kind_interp)

            # calculate the interpolated column and save it to the correct column of the data_inter array
            data_inter[:, col] = inter_func(time_inter)

    # check if time is not supposed to be shifted
    if not shift_time_axis:
        # shift back
        time_inter = time_inter * 1e9
        time_inter = time_inter + time_origin

    # return the interpolated time axis and data
    return time_inter, data_inter, sampling_rate


def _pad_android_data(sensor_data, report, start_with=None, end_with=None, padding_type='same'):
    """
    function in order to pad multiple android signals to the correct same start and end values. This function is
    needed in order to perform a synchronization of android sensors.
    Instead of passing a sample number or a time to indicate where to start or end the synchronization the user passes
    the sensor name instead.

    example:
    let's assume an acquisition was made with the following sensors Acc, GPS, Light, and Proximity.
    The sensors start acquiring data in the following order: ['Proximity', 'Light', 'Acc', 'GPS']
    The sensor stop acquiring data in the following order: ['Proximity', 'Light', 'GPS', 'Acc']

    Then the user can specify where to start and end the synchronization by for example setting:
    start_with='Proximity', and
    stop_with='GPS'
    In this case the signals are synchronised when the Proximity sensor starts recording until the GPS sensor stops
    recording data. The other sensors are padded / cropped to the corresponding starting / stopping points.
    At the beginning:The 'Light', 'Acc', and 'GPS' sensors are padded to the staring point of the Proximity sensor
    At the end: The 'Proximity' and 'Light' sensors are padded until the stopping point of the 'GPS' sensor and the
                'Acc' sensor is cropped to the stopping point of the GPS sensor.

    Parameters
    ----------
    sensor_data (list): A list containing the data of the sensors to be synchronised.

    report (dict): The report returned by the 'load_android_data' function.

    start_with (string, optional): The sensor that indicates when the synchronization should be started.
                              If not specified the sensor that started latest is chosen.

    end_with (string, optional): The sensor that indicates when the synchronizing should be stopped.
                            If not specified the sensor that stopped earliest is chosen

    padding_type (string, optional): The padding type used for padding the signal. Options are either 'same' or 'zero'.
                                     If not specified, 'same' is used.

    Returns
    -------

    padded_sensor_data: the padded sensor data for each sensor in the sensor_data list
    """

    # list for holding the padded data
    padded_sensor_data = []

    # get the index of the sensor used for padding in the start (ssi = start sensor index)
    # if none is provided (start == None) then the latest starting sensor is used
    if (start_with == None):
        ssi = report['starting times'].index(max(report['starting times']))

    else:
        ssi = report['names'].index(start_with)

    # get the index of the sensor used for padding in the end (esi = end sensor index)
    # if none is provided (end == None) then the sensor that stopped earliest is used
    if (end_with == None):
        esi = report['stopping times'].index(min(report['stopping times']))

    else:
        esi = report['names'].index(end_with)

    # check if the starting and stopping times are equal (this can be the case when a significant motion sensor is used
    # and only one significant motion was detected by the sensor)
    # in that case we use the next sensor that stopped recording the earliest
    if (report['starting times'][esi] == report['stopping times'][ssi]):
        print('Warning: Start and end at same time...using next sensor that stopped earliest instead')
        esi = report['stopping times'].index(np.sort(report['stopping times'])[1])

    # get the starting value
    start_time = report['starting times'][ssi]

    # get the stopping value
    end_time = report['stopping times'][esi]

    # get time axis of the starting sensor and check for dimensionality of the data
    time_axis_start = sensor_data[ssi][:1] if (sensor_data[ssi].ndim == 1) else sensor_data[ssi][:, 0]

    # get the time axis of th ending sensor and check for dimensionality of the data
    time_axis_end = sensor_data[esi][:1] if (sensor_data[esi].ndim == 1) else sensor_data[esi][:, 0]

    # start padding: for loop over names (enumerated)
    for i, name in enumerate(report['names']):

        # get the data of the current signal
        data = sensor_data[i]

        # check for the dimensionality of the signal data (handling for significant motion sensor)
        if (data.ndim == 1):  # 1D array

            # get the time axis
            time_axis = data[:1]

            # get the signal data
            signals = data[1:]

            # expand the dimensionality of the data (in order to have the same dimensionality as all other data)
            signals = np.expand_dims(signals, axis=1)

        else:  # mutlidimensional array

            # get the time_axis
            time_axis = data[:, 0]

            # get the signal data
            signals = data[:, 1:]

        # --- 1.) padding at the beginnging ---
        if (start_time > time_axis[0]):  # start_time after current signal start (cropping of the signal needed)

            # get the time_axis size before cropping
            orig_size = time_axis.size

            # crop the time axis
            time_axis = time_axis[time_axis >= start_time]

            # crop the signal data
            signals = signals[(orig_size - time_axis.size):, :]

        # get the values that need to be padded to the current time axis
        start_pad = time_axis_start[time_axis_start < time_axis[0]]

        # --- 2.) padding at the end ---
        if (end_time < time_axis[-1]):  # end_time before current signal end (cropping of the signal needed

            # crop the time axis
            time_axis = time_axis[time_axis <= end_time]

            # check if cropping leads to elimination of signal
            if (time_axis.size == 0):
                raise IOError(
                    'The configuration you chose led to elimination of the {} sensor. Please choose another sensor for paremeter \'end_with\'.'.format(
                        name))

            # crop the signal data
            signals = signals[:time_axis.size, :]

        # get the values that need to be padded to the current time axis
        end_pad = time_axis_end[time_axis_end > time_axis[-1]]

        # pad the time axis
        time_axis = np.concatenate((start_pad, time_axis, end_pad))

        # for holing the new padded data
        padded_data = time_axis

        # cycle over the signal channels
        for channel in np.arange(signals.shape[1]):

            # get the signal channel
            sig_channel = signals[:, channel]

            # check for the sensor
            if (name == 'GPS'):  # gps sensor (always use padding type 'same' to indicate that the phone has not moved)

                # pad the channel
                sig_channel = np.pad(sig_channel, (start_pad.size, end_pad.size), 'edge')

            elif (name == 'SigMotion'):  # significant motion sensor (always pad zeros)

                # pad the channel
                sig_channel = np.pad(sig_channel, (start_pad.size, end_pad.size), 'constant', constant_values=(0, 0))

            else:  # all other sensors

                # check for setting of the user
                if (padding_type == 'same'):

                    # pad the channel
                    sig_channel = np.pad(sig_channel, (start_pad.size, end_pad.size), 'edge')

                elif (padding_type == 'zeros'):

                    # pad the channel
                    sig_channel = np.pad(sig_channel, (start_pad.size, end_pad.size), 'constant',
                                         constant_values=(0, 0))

            # concatenate the channel to the padded data
            padded_data = np.vstack((padded_data, sig_channel))

        # append the data to the padded_sensor_data list
        # the data is transposed in order to have the correct shape (samples x number of channels)
        padded_sensor_data.append(padded_data.T)

    return padded_sensor_data


def _save_synchronised_data(time_axis, data, path, file_name='android_synchronized'):
    """
    Function used for saving synchronised android data into a single file. CHANGE DOCSTRING

    Parameters
    ----------
    time_axis (N,  array_like): The time axis after the padding and re-sampling the sensor data.

    data (list): List containing the padded and re-sampled sensor signals. The length of data along the 0-axis has to be
                 the same size as time_axis


    path (string): A string with the location where the file should be saved.

    file_name (string, optional): The name of the file, with the suffix '.txt'. If not specified, the file is named
                             'android_synchronised.txt'.

    """
    # add .txt suffix to the file name
    file_name = file_name + '.csv'
    # create final save path
    save_path = os.path.join(path, file_name)

    # add the time axis for the final data array
    # make the time axis a column vector
    final_data_array = np.expand_dims(time_axis, 1)

    # write all the data into a single array
    for signals in data:
        final_data_array = np.append(final_data_array, signals, axis=1)

    return save_path, final_data_array


def _sync_sensors_in_device(in_path, out_path, sync_file_name='android_synchronized') -> pd.DataFrame:
    """
    Function to synchronise multiple android files into one.

    Parameters
    ----------
    in_path (list): list of paths that point to the files that are supposed to be synchronised
    out_path (string): The path where the synchronised file is supposed to be saved
    sync_file_name (String, optional): The name of the new file. If not provided then the name will be set to
                                       'android_synchronised.txt'
    automatic_sync (boolean, optional): boolean for setting the mode of the function.
                                        If not provided it will be set to True

    Returns
    -------
    pandas DataFrame containing synchronised data from chosen sensors in device
    """

    # load the data
    sensor_data, report = load_device_data(in_path, print_report=True)

    # ---- data padding ---- #

    # inform the user
    print('Synchronizing from start of {} sensor until end of {} sensor.'.format(report['starting order'][-1],
                                                                                 report['stopping order'][0]))
    print('Using padding type: same.')

    padded_sensor_data = _pad_android_data(sensor_data, report)

    # ---- data re-sampling ---- #
    print('\n---- DATA RE-SAMPLING ----\n')

    # list for holding the re-sampled data
    re_sampled_data = []

    # list for holding the time axes of each sensor
    re_sampled_time = []

    # get the highest sampling rate and round it accordingly
    sampling_rate = round_sampling_rate(report['max. sampling rate'])

    # cycle over the sig
    for data in padded_sensor_data:
        # resample the data ('_' suppresses the output for the sampling rate)
        re_time, re_data, _ = _re_sample_data(data[:, 0], data[:, 1:], shift_time_axis=True,
                                              sampling_rate=sampling_rate, kind_interp='quadratic')

        # add the the time and data to the lists
        re_sampled_time.append(re_time)
        re_sampled_data.append(re_data)

    column_names = report['column names']

    # get save path and all data in one numoy array
    save_path, final_data_array = _save_synchronised_data(re_sampled_time[0], re_sampled_data, out_path,
                                                          file_name=sync_file_name)

    # put data into a pandas dataframe
    df = pd.DataFrame(final_data_array, columns=column_names)

    return df


def _extract_date_time(file_path: str) -> Tuple[str, str]:
    """
    extracts the date and the time from the file path

    Parameters
    ----------

    file_path: str
        Path to the file containing the data.

    Returns
    -------
    The time and the date strings
    """
    # get the date and the time from the path string
    date_time = file_path.rsplit('.', 1)[0].rsplit('_', 2)

    # extract date and time
    date = date_time[1]
    time = date_time[2]

    return date, time


def _get_sensor_path_list(folder_path: str, device: str, sensor_list: List[str]) -> List[str]:
    """
    Generates a list of file paths for sensor data based on specified device and sensor types.

    Parameters:
    ----------
    folder_path : str
        The directory path where sensor data files are located.

    device : str
        The type of device the data is being collected from. Supported devices are: "phone", "watch".

    sensor_list : List[str]
        A list of sensor types.

    Returns:
    -------
    A list containing the full paths to the files that match the specified device and
    sensor types.
    """
    sensor_path_list = []

    for sensor in sensor_list:
        sensor_name = _get_sensor_names(device, sensor)

        for filename in os.listdir(folder_path):

            if filename.startswith(sensor_name):
                sensor_path = os.path.join(folder_path, filename)

                # path for the data of the chosen sensors
                sensor_path_list.append(sensor_path)

    return sensor_path_list


def _sync_all_sensors_in_class(folder_path: str, out_path: str,
                               selected_sensors: Dict[str, List[str]], prefix: str = 'Sara') -> None:
    """
    Load data of chosen sensors for one class of movement.

    Parameters
    ----------
    data_path: str
        Path to the main directory containing the data.

    out_path : str
        The path where synchronized data will be saved.

    folder_name: str
        Folder name (class of movement) from which to load the data.

    selected_sensors: Dict[str, List[str]]
        A dictionary specifying devices as keys and a list of sensors to load data for as values.
        Supported sensor names for devices ("phone", "watch") include:
            "acc": accelerometer
            "gyr": gyroscope
            "mag": magnetometer
            "rotvec": rotation vector
            "wearheartrate": heart rate (watch only)
            "noise": ambient noise (phone only)

    prefix: str
        Prefix of the new file name

    Returns
    -------
    None: The function saves the synchronized data to a CSV file.
    """
    # get folder name from folder path
    folder_name = get_folder_name_from_path(folder_path)

    for device, sensor_list in selected_sensors.items():
        sensor_path_list = _get_sensor_path_list(folder_path, device, sensor_list)

        # extract date and time from the filenames - assume that sensors from the same device
        # have the dame date and time in the filename
        path_date_time = sensor_path_list[0]

        date, time = _extract_date_time(path_date_time)

        # generate file name
        sync_file_name = prefix + "_synchronized_" + device + '_' + folder_name + "_" + date + "_" + time + ".csv"

        # generate output directory
        output_path = create_dir(out_path, folder_name)

        # sync sensors from device
        df = _sync_sensors_in_device(sensor_path_list, output_path, sync_file_name)

        # save dataframe to a csv file
        df.to_csv(os.path.join(output_path, sync_file_name))
