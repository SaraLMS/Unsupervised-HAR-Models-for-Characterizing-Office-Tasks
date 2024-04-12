from typing import Dict, List

from synchronization.synchronization import synchronization
from visualization.visualize_sync import visualize_sync_signals

#
raw_data_in_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/signals_P001"
sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/sync_android_P001"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/synchronized_P001"
selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
sync_type = 'crosscorr'

evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/sync_evaluation_P001"
synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)

# file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/test_P001/synchronized_P001/walking_1/P001_synchronized_phone_watch_walking_1_2024-04-08_16_44_18_crosscorr.csv"
# visualize_sync_signals(file_path)
#

