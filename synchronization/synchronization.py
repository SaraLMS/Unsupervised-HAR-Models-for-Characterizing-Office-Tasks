# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import shutil
from typing import Dict, List
import os

from synchronization.sync_android_sensors import sync_all_classes
from synchronization.sync_devices_crosscorr import sync_crosscorr
from synchronization.sync_devices_timestamps import sync_timestamps


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def synchronization(raw_data_in_path: str, sync_android_out_path: str, selected_sensors: Dict[str, List[str]],
                    output_path:str, sync_type:str, save_intermediate_files: bool = True) -> None:
    # synchronize android sensors
    sync_all_classes(raw_data_in_path, sync_android_out_path, selected_sensors)

    # synchronize data from different devices
    for sync_folder, raw_folder in zip(os.listdir(sync_android_out_path), os.listdir(raw_data_in_path)):

        # get raw and synchronized folder paths
        sync_folder_path = os.path.join(sync_android_out_path, sync_folder)
        raw_folder_path = os.path.join(raw_data_in_path, raw_folder)


        if sync_type == 'crosscorr':
            # synchronize data based on cross correlation
            sync_crosscorr(sync_folder_path, output_path)

            # inform user
            print("Signals synchronized based on cross correlation")

        elif sync_type == 'timestamps':
            # synchronize data based on timestamps
            sync_timestamps(raw_folder_path, sync_folder_path, output_path)

            # inform user
            print("Signals synchronized based on timestamps")

    if not save_intermediate_files:
        # remove the folder containing the csv files generated when synchronizing android sensors
        shutil.rmtree(sync_android_out_path)


