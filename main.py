from typing import Dict, List

from synchronization.synchronization import synchronization
from visualization.visualize_sync import visualize_sync_signals


raw_data_in_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/jumps_acquisitions"
sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_data"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_devices_data"
selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
sync_type = 'timestamps'

evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/sync_eval"
synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)


# # visualize_sync_signals(file_path)
#
# from synchronization.sync_evaluation import sync_evaluation
#
# report = sync_evaluation(logger_folder_path, folder_path)
# print(report)
