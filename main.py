from synchronization.sync_devices_crosscorr import sync_crosscorr
from synchronization.sync_devices_timestamps import sync_on_filename_timestamps

folder_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/synchronized_data/cabinets_1"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/introdutory_task/test"
sync_on_filename_timestamps(folder_path, output_path)