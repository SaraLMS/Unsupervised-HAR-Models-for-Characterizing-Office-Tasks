from feature_engineering.feature_selection import feature_selector
from feature_extraction.feature_extraction import feature_extractor, generate_cfg_file
from processing.processor import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals


#
# #

# raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P006"
# sync_android_out_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/sync_android_P006"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/synchronized_P006"
# selected_sensors = {'phone': ['acc', 'gyr', 'mag']}
# sync_type = 'crosscorr'
# #
# evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/sync_evaluation_P006"
# synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,evaluation_path)
# #
# # # file_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P001/synchronized_P001/standing_2/P001_synchronized_phone_watch_standing_2_2024-04-11_11_37_30_crosscorr.csv"
# # # visualize_sync_signals(file_path)
# # # #
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone"
# filtered_folder_name = "filtered_tasks_P006"
# raw_folder_name = "raw_tasks_P006"
# sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/sync_android_P006"
# # cut and filter
# processing(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

# generate cfg file
# path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
# generate_cfg_file(path)
# # #
# subclasses = ['standing_still', 'walk_medium', 'sit']
# main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/filtered_tasks_P006"
# output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone"
# feature_extractor(main_path, output_path, subclasses)


# path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone/features/acc_gyr_mag_phone_features_P006.csv"
# df = load_data_from_csv(path)
# output_path_plots = "C:/Users/srale/OneDrive - FCT NOVA/Tese/P006/acc_gyr_mag_phone"
# feature_selector(df, 8, "KMeans", output_path_plots)
#
def main():
    # Set these booleans to True or False depending on which steps you want to run
    do_synchronization = False
    do_processing = True
    do_generate_cfg_file = False
    do_feature_extraction = True
    do_feature_selection = True

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P004"
        sync_android_out_path = ("C:/Users/srale/OneDrive - FCT "
                                 "NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_android_P004")
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/synchronized_P004"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_evaluation_P004"
        synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,
                        evaluation_path)

    if do_processing:
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
        filtered_folder_name = "filtered_tasks_P004"
        raw_folder_name = "raw_tasks_P004"
        sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_android_P004"
        processing(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/filtered_tasks_P004"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
        feature_extractor(main_path, output_path, subclasses)

    if do_feature_selection:
        dataset_path = ("C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/features_basic_activities"
                        "/acc_gyr_mag_phone_features_P004.csv")
        output_path_plots = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
        clustering_model = "KMeans"
        nr_iterations = 20
        feature_sets = feature_selector(dataset_path, nr_iterations, clustering_model, output_path_plots)
        print(feature_sets)


if __name__ == "__main__":
    main()
