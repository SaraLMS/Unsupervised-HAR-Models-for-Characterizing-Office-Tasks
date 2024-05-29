from typing import Dict, List

from feature_extraction.feature_extractor import feature_extractor, generate_cfg_file
from processing.processor import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals
#
# #
# raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P008"
# sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P008/acc_gyr_mag_phone_and_watch/sync_android_P008"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P008/acc_gyr_mag_phone_and_watch/synchronized_P008"
# selected_sensors = {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}
# sync_type = 'crosscorr'
# # #
# evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P008/acc_gyr_mag_phone_and_watch"
# synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)
#
# # file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001/standing_2/P001_synchronized_phone_watch_standing_2_2024-04-11_11_37_30_crosscorr.csv"
# # visualize_sync_signals(file_path)
# # #
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P008/acc_gyr_mag_phone_and_watch"
# filtered_folder_name = "filtered_tasks_P008"
# raw_folder_name = "raw_tasks_P008"
# sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P008/acc_gyr_mag_phone_and_watch/synchronized_P008"
# # cut and filter
# processing(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

# generate cfg file
# path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
# generate_cfg_file(path)
# # #
main_path = "D:/tese_backups/raw_signals_backups/processed/P001/acc_gyr_mag_phone_and_watch/filtered_tasks_P001"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/acc_gyr_mag_phone_and_watch/features"

feature_extractor(main_path, output_path)



