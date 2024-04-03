# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import shutil
from glob import glob
from typing import Dict, List
import os

import pandas as pd

from constants import CROSSCORR, TIMESTAMPS
from synchronization.sync_android_sensors import sync_all_classes
from synchronization.sync_devices_crosscorr import sync_crosscorr
from synchronization.sync_devices_timestamps import sync_timestamps
from synchronization.sync_evaluation import sync_evaluation


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def synchronization(raw_data_in_path: str, sync_android_out_path: str, selected_sensors: Dict[str, List[str]],
                    output_path: str, sync_type: str, evaluation_output_path: str,
                    save_intermediate_files: bool = True) -> None:
    """
    Synchronizes android sensor data and between two different devices. Two different synchronization methods are
    supported: cross correlation and timestamps. Generates a new csv file containing all the synchronized sensor data
    from the two devices.
    MuscleBans not entirely implemented.

    Parameters:
        raw_data_in_path (str):
        Main folder path containing subfolders with raw sensor data.

        sync_android_out_path (str):
        Path to the location where the synchronized android sensor data will be saved

        selected_sensors (Dict[str, List[str]):
        Dictionary containing the devices and sensors chosen to be loaded and synchronized.
        Devices supported are:
            "watch": smartwatch
            "phone": smartphone
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
        Path to save the synchronization evaluation

        save_intermediate_files (bool): Default = True
        Keep the csv files generated when synchronizing android sensors. False to delete.
    """
    # check if in path is valid
    _check_in_path(raw_data_in_path)

    # check if selected sensors are valid
    _check_supported_sensors(selected_sensors)

    # check if accelerometer is selected for every device
    _check_acc_sensor_selected(selected_sensors)

    # synchronize android sensors
    sync_all_classes(raw_data_in_path, sync_android_out_path, selected_sensors)

    # array for holding the dataframes containing the sync evaluation
    evaluation_df_array = []

    # synchronize data from different devices
    for sync_folder, raw_folder in zip(os.listdir(sync_android_out_path), os.listdir(raw_data_in_path)):

        # get raw and synchronized folder paths
        sync_folder_path = os.path.join(sync_android_out_path, sync_folder)
        raw_folder_path = os.path.join(raw_data_in_path, raw_folder)

        if sync_type == CROSSCORR:
            # synchronize data based on cross correlation
            sync_crosscorr(sync_folder_path, output_path)

            # inform user
            print("Signals synchronized based on cross correlation")

        elif sync_type == TIMESTAMPS:
            # synchronize data based on timestamps
            sync_timestamps(raw_folder_path, sync_folder_path, output_path)

            # inform user
            print("Signals synchronized based on timestamps")

        sync_report_df = sync_evaluation(raw_folder_path, sync_folder_path)
        evaluation_df_array.append(sync_report_df)

    if not save_intermediate_files:
        # remove the folder containing the csv files generated when synchronizing android sensors
        shutil.rmtree(sync_android_out_path)

    # concat dataframes in array to one
    combined_df = pd.concat(evaluation_df_array, ignore_index=True)

    # define filename
    filename = "evaluation_report.csv"

    # define output path
    evaluation_output_path = os.path.join(evaluation_output_path, filename)

    # save csv file containing sync evaluation
    combined_df.to_csv(evaluation_output_path)


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

    supported_devices = ["watch", "phone"]
    supported_sensors = {
        "watch": ["acc", "gyr", "mag", "rotvec", "wearheartrate"],
        "phone": ["acc", "gyr", "mag", "rotvec", "noise"]
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


def _check_in_path(raw_data_in_path: str) -> None:
    """
    Checks if the specified path is valid according to the criteria:
    - The path exists and is a directory.
    - Contains subdirectories.
    - Each subdirectory contains at least one .txt file.

    Parameters:
    raw_data_in_path (str):
    The main folder path containing subfolders with raw sensor data.

    Raises:
    ValueError: If any of the criteria are not met.
    """
    if not os.path.isdir(raw_data_in_path):
        raise ValueError(f"The path {raw_data_in_path} does not exist or is not a directory.")

    subfolders = [f.path for f in os.scandir(raw_data_in_path) if f.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in the main path {raw_data_in_path}.")

    for subfolder in subfolders:
        txt_files = glob(os.path.join(subfolder, "*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in subfolder {subfolder}.")


def _check_acc_sensor_selected(selected_sensors: Dict[str, List[str]]) -> None:
    """
    Checks if the accelerometer ("acc") sensor is selected for each device.

    Parameters:
    selected_sensors (Dict[str, List[str]]):
    Dictionary containing the devices and sensors chosen to be loaded and synchronized.

    Raises:
    ValueError: If the accelerometer sensor is not selected for any of the devices, listing all such devices.
    """
    devices_missing_acc = [device for device, sensors in selected_sensors.items() if "acc" not in sensors]

    if devices_missing_acc:
        missing_str = ", ".join(devices_missing_acc)
        raise ValueError(f"Accelerometer sensor ('acc') must be selected for device(s): {missing_str}")