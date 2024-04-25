from typing import Dict, List

from processing.processing import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals

#
raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/signals_P002"
sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/sync_android_P002"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/synchronized_P002"
selected_sensors = {'phone': ['acc', 'gyr'], 'watch': ['acc']}
sync_type = 'crosscorr'

evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/sync_evaluation_P002"
synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)

# file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001/standing_2/P001_synchronized_phone_watch_standing_2_2024-04-11_11_37_30_crosscorr.csv"
# visualize_sync_signals(file_path)


# sync_data_main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001"
#
# filtered_data_dict = processing(sync_data_main_path)
#
# # visualize_filtering_results(filtered_data_dict, sync_data_main_path)
# df = filtered_data_dict['cabinets_1']



# import matplotlib.pyplot as plt
# plt.plot(df['yAcc'], label='y-axis acc phone')
# plt.show()