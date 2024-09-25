# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import shutil
import re
from typing import Dict, List
import os
import parser
import pandas as pd

# internal imports
from constants import CROSSCORR, TIMESTAMPS, ACCELEROMETER, WEAR_ACCELEROMETER, WATCH, PHONE, SUPPORTED_DEVICES, MBAN, \
    SUPPORTED_PHONE_SENSORS, SUPPORTED_WATCH_SENSORS, SUPPORTED_MBAN_SENSORS, ACC, TXT, CSV
from .sync_android_sensors import sync_all_classes
from .sync_devices_crosscorr import sync_crosscorr
from .sync_devices_timestamps import sync_timestamps
from .evaluation import sync_evaluation


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def synchronize_all(main_folder_in_path: str, selected_sensors: Dict[str, List[str]], sync_type: str,
                    output_base_path: str, sync_android_output_filename: str = "sync_android_sensors",
                    sync_devices_output_filename: str = "sync_devices") -> None:
    """
    Synchronizes all signals present in a main directory. Check synchronization function bellow for more details.

    :param main_folder_in_path: str
    Path to the main folder containing subfolders with raw sensor data (i.e., ../main_folder/subfolders/sensor_data.txt)
    The sub folders should have the following structure: four characters, starting with an upper case letter, and
    followed by three digits (i.e., P001)

    :param selected_sensors: Dict[str, List[str]
    Dictionary containing the devices and sensors to be loaded and synchronized.

    :param sync_type: str
    Method for synchronizing data between devices.

    :param output_base_path: str
    Path to the location where the file containing the synchronized data should be saved.

    :param sync_android_output_filename: str
    Name of the folder where to store the csv files from the synchronization of Android sensors within a device.

    :param sync_devices_output_filename: str
    Name of the folder where to store the csv files from the synchronization between devices.

    :return: None
    """

    # check if selected sensors are valid
    _check_supported_sensors(selected_sensors)

    # check if the MuscleBAN was selected
    _check_if_mban_selected(selected_sensors)

    # check if ACC sensor is selected for both devices if cross correlation is chosen
    _check_if_acc_selected_for_crosscorr(selected_sensors, sync_type)

    # Compile the regular expression for valid subfolder names
    pattern = re.compile(r'^[A-Z]\d{3}$')

    for sub_folder in os.listdir(main_folder_in_path):

        if pattern.match(sub_folder):
            raw_data_in_path = os.path.join(main_folder_in_path, sub_folder)

            synchronization(raw_data_in_path, selected_sensors, output_base_path, sync_android_output_filename,
                            sync_devices_output_filename, sync_type, sub_folder)


def synchronization(raw_data_in_path: str, selected_sensors: Dict[str, List[str]], output_base_path: str,
                    sync_android_output_filename: str, sync_devices_output_filename: str, sync_type: str,
                    sub_folder_name: str, evaluation_filename: str = "evaluation_report",
                    save_intermediate_files: bool = True) -> None:
    """
    Synchronizes android sensor data and between two different devices. Two different synchronization methods are
    supported: cross correlation and timestamps. Generates a new csv file containing all the synchronized sensor data
    from the two devices.
    MuscleBans not entirely implemented. Only smartwatch and smartphone.

    To synchronize the signals based on cross correlation, every acquisition must start with a series of vertical
    jumps with the arms straight and parallel to the trunk. When choosing cross correlation, the window (in samples)
    containing the jumps should be correctly defined in _get_axis_from_acc in synchronization.sync_devices_crosscorr.
    Without the correct window_size the results might be incorrect.

    Synchronization based on time stamps extracts the start times of the sensors from the logger .txt file. If
    this file does not exist or does not contain the needed start times, the start times in the raw data filenames will
    be used.

    Generates a csv file containing the evaluation of the performance of the three synchronization methods:
    cross correlation and timestamps (logger file and filename start times); being the cross correlation the
    reference method since, if followed the jumps protocol and with the correct window size, presents the best results.

    :param raw_data_in_path: str
    Path to the sub folder containing other sub_folders with raw sensor data
    (i.e., ../main_folder/sub folder/sub_folders/sensor_data.txt)

    :param selected_sensors: Dict[str, List[str]
    Dictionary containing the devices and sensors to be loaded and synchronized.
    Devices supported are:
        "watch": smartwatch
        "phone": smartphone
        "mban" : MuscleBAN - To implement later
    Supported sensors include:
        "acc": accelerometer
        "gyr": gyroscope
        "mag": magnetometer
        "rotvec": rotation vector
        "wearheartrate": heart rate (watch only)
        "noise": ambient noise (phone only)

    :param output_base_path: str
    Path to the location where the file containing the synchronized data should be saved.

    :param sync_android_output_filename: str
    Name of the folder where to store the csv files from the synchronization of Android sensors within a device.

    :param sync_devices_output_filename: str
    Name of the folder where to store the csv files from the synchronization between devices.

    :param sync_type: str
    Method for synchronizing data between devices. Supported methods:
        "crosscorr": Cross correlation between the ACC signals from the device
        (y-axis for the phone and -x-axis for the watch)

        "timestamps": Start times in the logger file. If this file does not exist or does not have the needed timestamps
        the start times in the raw data filenames will be used instead.

    :param sub_folder_name: str
    Name of the sub folder containing the raw data files (.txt)

    :param evaluation_filename: str (default = evalution_report)
    Name of the file which will contain the synchronization evaluation report.

    :param save_intermediate_files: bool (default = True)
    Keep the csv files generated after synchronizing android sensors. False to delete.
    If there's only signals from one device, these files are not deleted.

    :return: None
    """

    # TODO check if the chosen sensors exist!
    # check if in path is valid and contains txt files inside
    parser.check_in_path(raw_data_in_path, TXT)

    # generate folder name with the sensors and devices loaded
    sensors_devices_foldername = _generate_folder_name_based_on_selected_sensors(selected_sensors)

    # generate output path for the synchronized android sensors
    sync_android_out_path = os.path.join(output_base_path, sub_folder_name, sensors_devices_foldername, sync_android_output_filename)

    # generate output path for the synchronized devices
    sync_devices_output_path = os.path.join(output_base_path, sub_folder_name, sensors_devices_foldername, sync_devices_output_filename)

    # inform user
    print(f"Synchronizing signals from {sub_folder_name}")

    # synchronize android sensors
    # if there's only one device, sync android sensors and save csv files
    sync_all_classes(sub_folder_name, raw_data_in_path, sync_android_out_path, selected_sensors)

    # synchronize in pairs of devices
    if len(selected_sensors) == 2:

        # array for holding the dataframes containing the synchronization results
        evaluation_df_array = []

        # synchronize data from different devices
        for sync_folder, raw_folder in zip(os.listdir(sync_android_out_path), os.listdir(raw_data_in_path)):

            # get raw and synchronized folder paths
            sync_folder_path = os.path.join(sync_android_out_path, sync_folder)
            raw_folder_path = os.path.join(raw_data_in_path, raw_folder)

            if sync_type == CROSSCORR:

                # check if accelerometer files exist
                _check_acc_file(raw_folder_path, selected_sensors)

                # synchronize data based on cross correlation
                sync_crosscorr(sub_folder_name, sync_folder_path, sync_devices_output_path)

                # inform user
                print("Signals synchronized based on cross correlation")

            elif sync_type == TIMESTAMPS:
                # synchronize data based on timestamps
                sync_timestamps(sub_folder_name, raw_folder_path, sync_folder_path, sync_devices_output_path,
                                selected_sensors)

                # inform user
                print("Signals synchronized based on timestamps")

            sync_report_df = sync_evaluation(raw_folder_path, sync_folder_path, selected_sensors)
            evaluation_df_array.append(sync_report_df)

        if not save_intermediate_files:
            # remove the folder containing the csv files generated when synchronizing android sensors
            shutil.rmtree(sync_android_out_path)

        # concat dataframes in array to one
        combined_df = pd.concat(evaluation_df_array, ignore_index=True)

        # add csv suffix
        evaluation_filename = f"{evaluation_filename}{CSV}"

        # define output path
        evaluation_output_path = os.path.join(output_base_path, sub_folder_name, sensors_devices_foldername, evaluation_filename)

        # save csv file containing sync evaluation
        combined_df.to_csv(evaluation_output_path, index=False)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _check_supported_sensors(selected_sensors: Dict[str, List[str]]):
    """
    Check if the selected sensors are supported for the chosen devices
    and if the sensors and devices chosen are valid.

    :param selected_sensors: Dict[str, List[str]]
    Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    :return: None
    """
    supported_devices = SUPPORTED_DEVICES
    supported_sensors = {
        PHONE: SUPPORTED_PHONE_SENSORS,
        WATCH: SUPPORTED_WATCH_SENSORS,
        MBAN: SUPPORTED_MBAN_SENSORS
    }

    invalid_devices = [device for device in selected_sensors.keys() if device not in supported_devices]
    unsupported_sensors_info = {}

    for device, sensors in selected_sensors.items():
        invalid_sensors = [sensor for sensor in sensors if sensor not in supported_sensors.get(device, [])]
        if invalid_sensors:
            unsupported_sensors_info.setdefault(device, []).extend(invalid_sensors)

    error_messages = []

    if invalid_devices:
        error_messages.append(f"The following device names are invalid: {', '.join(invalid_devices)}")

    for device, unsupported_sensors in unsupported_sensors_info.items():
        error_messages.append(
            f"The following sensors are not supported for device '{device}': {', '.join(unsupported_sensors)}")

    if error_messages:
        raise ValueError("\n".join(error_messages))


def _check_acc_sensor_selected(selected_sensors: Dict[str, List[str]]) -> None:
    """
    Checks if the accelerometer (acc) sensor is selected for each device.

    :param selected_sensors: Dict[str, List[str]]:
    Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    :return: None
    """
    devices_missing_acc = [device for device, sensors in selected_sensors.items() if ACC not in sensors]

    if devices_missing_acc:
        missing_str = ", ".join(devices_missing_acc)
        raise ValueError(f"Accelerometer sensor ('acc') must be selected for device(s): {missing_str}")


def _check_acc_file(folder_path: str, selected_sensors: Dict[str, List[str]]) -> None:
    """
    Checks for the presence of accelerometer data files for each selected device in the specified folder. Raises an
    error if accelerometer data is missing for any device selected for synchronization.

    :param folder_path: str
    Path to the folder containing the sensor data files (.txt files)

    :param selected_sensors: Dict[str, List[str]]
    A dictionary where keys are device types and values are lists of sensors selected for each device type.

    :return: None
    """
    # Map device types to the expected substring in the accelerometer file names
    acc_file_identifiers = {
        WATCH: WEAR_ACCELEROMETER,
        PHONE: ACCELEROMETER
        # Add other device types
    }

    # Identify which devices are supposed to have accelerometer data
    devices_needing_acc_files = [device for device, sensors in selected_sensors.items() if ACC in sensors]

    # Keep track of which devices have accelerometer data
    verified_devices = set()

    # Check files in the folder
    for filename in os.listdir(folder_path):
        for device in devices_needing_acc_files:
            if acc_file_identifiers[device] in filename.upper():
                verified_devices.add(device)

    # Determine if any selected devices are missing accelerometer data
    missing_devices = set(devices_needing_acc_files) - verified_devices

    if missing_devices:
        missing_str = ", ".join(missing_devices)
        raise ValueError(f"Missing accelerometer data file(s) for device(s): {missing_str}")


def _check_if_mban_selected(selected_sensors: Dict[str, List[str]]) -> None:
    """
    Check if the MuscleBAN device was chosen to load and synchronize. If so, raise a Value Error exception.

    :param selected_sensors:  Dict[str, List[str]]
    A dictionary where keys are device types and values are lists of sensors

    :return: None
    """

    # check if the mban is in the selected sensors
    if MBAN in selected_sensors:
        # if the user chooses the mban raise exception
        raise ValueError("The device 'MuscleBAN' has not been implemented.")


def _check_if_acc_selected_for_crosscorr(selected_sensors: Dict[str, List[str]], sync_type: str) -> None:
    """
    Check if the acceleromenter sensor was selected for both devices if the synchronization method chosen is cross
    correlation. If not, raise value error exception.

    :param selected_sensors: Dict[str, List[str]]
    A dictionary where keys are device types and values are lists of sensors

    :param sync_type: str
    Method for synchronizing data between devices. Supported methods:
        "crosscorr": Cross correlation:
        "timestamps": Start times in the sensor files

    :return: None
    """

    # check the sync type
    if sync_type == CROSSCORR:
        # if cross corr, check if the accelerometer from the two devices was chosen
        _check_acc_sensor_selected(selected_sensors)


def _generate_folder_name_based_on_selected_sensors(selected_sensors: Dict[str, List[str]]) -> str:
    """
    Generate a folder name based on selected sensors and their associated devices.

    The folder name is constructed by concatenating unique sensor names (in the order they are first encountered)
    from the provided dictionary, followed by the names of the devices. The sensor names and device names are
    separated by underscores.

    :param selected_sensors:
    A dictionary where keys are device names (str) and values are lists of sensor names (List[str]) selected for each
    device. Example: {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}

    :return: str
    A string representing the combined folder name in the format 'sensor1_sensor2_..._device1_device2_...'
    (e.g., 'acc_gyr_mag_phone_watch')
    """
    # Create a list to hold unique sensor names while preserving order
    sensor_list = []
    seen_sensors = set()  # To keep track of seen sensors

    # Iterate over the dictionary
    for sensors in selected_sensors.values():
        for sensor in sensors:
            if sensor not in seen_sensors:
                seen_sensors.add(sensor)
                sensor_list.append(sensor)

    # Create a string of unique sensors separated by underscores
    sensor_string = "_".join(sensor_list)

    # Add the device names to the end
    device_names = "_".join(selected_sensors.keys())

    # Combine the sensor string and device names
    final_string = f"{sensor_string}_{device_names}"

    return final_string
