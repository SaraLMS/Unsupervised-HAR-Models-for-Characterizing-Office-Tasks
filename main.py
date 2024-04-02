from synchronization.sync_devices_timestamps import sync_timestamps
from synchronization.synchronization import synchronization
from visualization.visualize_sync import visualize_sync_signals
# raw_data_in_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/jumps_acquisitions"
# sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_data"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_devices_data"
# selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
# sync_type = 'timestamps'
#
#
# synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type)



#
# logger_folder_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/jumps_acquisitions/jumps"
# folder_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_data/jumps"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/test"
#
# sync_timestamps(logger_folder_path, folder_path, output_path)

logger_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/test/jumps/Sara_synchronized_phone_watch_jumps_2024-03-13_14_45_01_logger_timestamps.csv"
file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/test/jumps/Sara_synchronized_phone_watch_jumps_2024-03-13_14_45_01_filename_timestamps.csv"

visualize_sync_signals(file_path)