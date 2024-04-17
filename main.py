from typing import Dict, List

from filtering.filtering import filtering
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals

#
# raw_data_in_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/signals_P001"
# sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/sync_android_P001"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001"
# selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
# sync_type = 'crosscorr'
#
# evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/sync_evaluation_P001"
# synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)

# file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/synchronized_P001/sitting_1/P001_synchronized_phone_watch_sitting_1_2024-04-08_17_56_28_crosscorr.csv"
# visualize_sync_signals(file_path)
#

sync_data_main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001"

filtered_data_dict = filtering(sync_data_main_path)

visualize_filtering_results(filtered_data_dict, sync_data_main_path)
