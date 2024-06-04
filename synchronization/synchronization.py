# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import shutil
from typing import Dict, List
import os

import pandas as pd

from constants import CROSSCORR, TIMESTAMPS, ACCELEROMETER, WEAR_ACCELEROMETER, WATCH, PHONE, SUPPORTED_DEVICES, MBAN, \
    SUPPORTED_PHONE_SENSORS, SUPPORTED_WATCH_SENSORS, SUPPORTED_MBAN_SENSORS, ACC
from parser.check_create_directories import check_in_path, create_dir
from synchronization.sync_android_sensors import sync_all_classes
from synchronization.sync_devices_crosscorr import sync_crosscorr
from synchronization.sync_devices_timestamps import sync_timestamps
from synchronization.sync_evaluation import sync_evaluation


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def synchronization(raw_data_in_path: str, sync_android_out_path: str, selected_sensors: Dict[str, List[str]],
                    output_path: str, sync_type: str, evaluation_output_path: str,
                    evaluation_filename: str = "evaluation_report_acc_gyr_phone.csv", save_intermediate_files: bool = True,
                    prefix: str = "P005") -> None:
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

    Parameters:
        prefix (str):
        Prefix do add to the generated filenames.

        raw_data_in_path (str):
        Main folder path containing subfolders with raw sensor data.

        sync_android_out_path (str):
        Path to the location where the synchronized android sensor data will be saved

        selected_sensors (Dict[str, List[str]):
        Dictionary containing the devices and sensors chosen to be loaded and synchronized.
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

        output_path (str):
        Path to the location where the file containing the synchronized data should be saved.

        sync_type (str):
        Method for synchronizing data between devices. Supported methods:
            "crosscorr": Cross correlation:
            "timestamps": Start times in the sensor files

        evaluation_output_path (str):
        Path to save the synchronization evaluation.

        evaluation_filename (str):
        Name of the file containing the sync evaluation report.

        save_intermediate_files (bool): Default = True.
        Keep the csv files generated after synchronizing android
        sensors. False to delete. If there's only signals from one device, these files are not deleted.
    """
    # check if in path is valid
    check_in_path(raw_data_in_path, '.txt')

    # check if selected sensors are valid
    _check_supported_sensors(selected_sensors)

    # TODO check if the chosen sensors exist!!!!!!!!!!!!!

    # TODO check if mban was selected - raise exception
    # synchronize android sensors
    # if there's only one device, sync android sensors and save csv files
    sync_all_classes(prefix, raw_data_in_path, sync_android_out_path, selected_sensors)

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
                # check if accelerometer is selected for every device
                _check_acc_sensor_selected(selected_sensors)

                # check if accelerometer files exist
                _check_acc_file(raw_folder_path, selected_sensors)

                # synchronize data based on cross correlation
                sync_crosscorr(prefix, sync_folder_path, output_path)

                # inform user
                print("Signals synchronized based on cross correlation")

            elif sync_type == TIMESTAMPS:
                # synchronize data based on timestamps
                sync_timestamps(prefix, raw_folder_path, sync_folder_path, output_path, selected_sensors)

                # inform user
                print("Signals synchronized based on timestamps")

            sync_report_df = sync_evaluation(raw_folder_path, sync_folder_path, selected_sensors)
            evaluation_df_array.append(sync_report_df)

        if not save_intermediate_files:
            # remove the folder containing the csv files generated when synchronizing android sensors
            shutil.rmtree(sync_android_out_path)

        # concat dataframes in array to one
        combined_df = pd.concat(evaluation_df_array, ignore_index=True)

        # create dir
        evaluation_output_path = create_dir(evaluation_output_path, folder_name="sync_evaluation_"+prefix)
        # define output path
        evaluation_output_path = os.path.join(evaluation_output_path, evaluation_filename)

        # save csv file containing sync evaluation
        combined_df.to_csv(evaluation_output_path, index=False)

    if len(selected_sensors) > 2:
        print("MuscleBANs not implemented yet. Available devices: phone and watch")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _check_supported_sensors(selected_sensors: Dict[str, List[str]]):
    """
    Check if the selected sensors are supported for the chosen devices
    and if the sensors and devices chosen are valid.

    Parameters:
    selected_sensors (Dict[str, List[str]]):
    Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    Raises:
    ValueError: If any selected sensor is not supported for the corresponding device,
    or if any device name or sensor name is invalid.
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
    Checks if the accelerometer ("acc") sensor is selected for each device.

    Parameters:
    selected_sensors (Dict[str, List[str]]):
    Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    Raises:
    ValueError: If the accelerometer sensor is not selected for any of the devices, listing all such devices.
    """
    devices_missing_acc = [device for device, sensors in selected_sensors.items() if ACC not in sensors]

    if devices_missing_acc:
        missing_str = ", ".join(devices_missing_acc)
        raise ValueError(f"Accelerometer sensor ('acc') must be selected for device(s): {missing_str}")


def _check_acc_file(folder_path: str, selected_sensors: Dict[str, List[str]]) -> None:
    """
    Checks for the presence of accelerometer data files for each selected device in the specified folder. Raises an
    error if accelerometer data is missing for any device selected for synchronization.

    Parameters:
    - folder_path (str): The path to the folder containing the data files.
    - selected_sensors (Dict[str, List[str]]): A dictionary where keys are device types and values are lists of sensors
      selected for each device type.

    Raises:
    - ValueError: If accelerometer data is missing for any of the selected devices.
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
