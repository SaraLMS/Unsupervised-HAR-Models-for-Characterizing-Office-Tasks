from typing import Dict, List

from feature_extraction.feature_extractor import feature_extractor, generate_cfg_file
from processing.processor import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals


# raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P002"
# sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/sync_android_P002"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch/sync"
# selected_sensors = {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}
# sync_type = 'crosscorr'
# #
# evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch/evaluation"
# synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)
#
# # file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001/standing_2/P001_synchronized_phone_watch_standing_2_2024-04-11_11_37_30_crosscorr.csv"
# # visualize_sync_signals(file_path)
# # #
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch/cut"
# sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch/sync"
# # cut and filter
# processing(sync_data_path, output_path)

# # #
main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch/cut"
output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P002/acc_gyr_phone_watch"
feature_extractor(main_path, output_path)



