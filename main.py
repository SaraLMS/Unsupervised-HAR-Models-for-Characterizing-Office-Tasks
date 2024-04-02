from synchronization.sync_devices_timestamps import sync_timestamps
from synchronization.synchronization import synchronization

raw_data_in_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/jumps_acquisitions"
sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_data"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_devices_data"
selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
sync_type = 'timestamps'


synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type)



