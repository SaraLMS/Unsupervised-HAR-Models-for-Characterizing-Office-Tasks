import feature_engineering
import processing
import synchronization
import clustering
from processing.pre_processing import _apply_filters
import load


# path ="D:/tese_backups/subjects/P016/acc_gyr_mag_phone/raw_tasks_P016/stairs/P016_synchronized_phone_stairs_2024-07-09_16-13-30_stairsdown2.csv"
# # load data to csv
# data = load.load_data_from_csv(path)
# # filter signals
# filtered_data = _apply_filters(data, 100)
#
# # cut first 200 samples to remove impulse response from the butterworth filters
# filtered_data = filtered_data.iloc[200:]
#
# path_filt = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone/filtered_tasks_P016/stairs/P016_synchronized_phone_stairs_2024-07-09_16-13-30_stairsdown2.csv"
#
# filtered_data.to_csv(path_filt)


def main():
    # Set these booleans to True or False depending on which steps to run
    do_synchronization = False
    do_processing = False
    do_generate_cfg_file = False
    do_feature_extraction = False
    do_one_subject_feature_selection = True
    do_general_model_feature_selection = False
    do_all_subjects_feature_selection = False
    do_general_model_clustering = False

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P016"
        sync_android_out_path = ("D:/tese_backups/subjects/P016/acc_gyr_mag_phone/sync_android_P016")
        output_path = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone/synchronized_P016"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        evaluation_path = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone"
        synchronization.synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path,
                                        sync_type, evaluation_path)

    if do_processing:
        output_path = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone"
        filtered_folder_name = "filtered_tasks_P016"
        raw_folder_name = "raw_tasks_P016"
        sync_data_path = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone/sync_android_P016"
        processing.processor(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        feature_engineering.generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']  # , 'standing_gestures', 'stairs'
        main_path = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone/filtered_tasks_P016"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P016"
        feature_engineering.feature_extractor(main_path, output_path, subclasses)

    if do_one_subject_feature_selection:
        dataset_path = (
            "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P016/phone_features_basic_activities"
            "/acc_gyr_mag_phone_features_P016.csv")
        #
        df = load.load_data_from_csv(dataset_path)

        # train test split
        train_set, _ = load.train_test_split(df, 0.8, 0.2)

        output_path_plots = "D:/tese_backups/subjects/P016/acc_gyr_mag_phone"
        _, _, _, _ = feature_engineering.feature_selector(train_set, 0.01, 0.99, 20,
                                                          "kmeans", output_path_plots)
        # results_output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/results"
        # results_folder_name = "subject_specific_feature_selection"
        # results_filename_prefix = "phone"
        # feature_engineering.one_stage_feature_selection(dataset_path, 0.1, 0.99,
        #                                                 35, "kmeans", output_path_plots, results_output_path,
        #                                                 results_folder_name, results_filename_prefix)

    if do_general_model_feature_selection:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        subfolder_name = "phone_features_basic_activities"
        output_path = "D:/tese_backups/general_model"
        # have to do a train_split??
        # train_subjects_df, _ = load.load_all_subjects(main_path, subfolder_name)
        # _, _ = feature_engineering.feature_selector(train_subjects_df, 0.01, 20, "kmeans", output_path)

    if do_all_subjects_feature_selection:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        features_folder_name = "watch_phone_features_basic_activities"
        clustering_model = "kmeans"
        feature_selection_iterations = 10
        variance_threshold = 0.05
        correlation_threshold = 0.99
        top_n = 4  # Number of top features to select
        nr_iterations = 15

        feature_engineering.two_stage_feature_selection(main_path, features_folder_name, variance_threshold,
                                                        correlation_threshold,
                                                        feature_selection_iterations, clustering_model, nr_iterations,
                                                        top_n)

    if do_general_model_clustering:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        subfolder_name = "phone_features_basic_activities"
        clustering_model = "kmeans"
        feature_set = ['yMag_Max', 'xMag_Spectral centroid', 'yAcc_Interquartile range', 'zAcc_Interquartile range',
                       'zGyr_Interquartile range', 'zAcc_Min', 'yGyr_Spectral entropy']
        _, _, _ = clustering.general_model_clustering(main_path, subfolder_name, clustering_model, feature_set)


if __name__ == "__main__":
    main()

# import pandas as pd
#
# path_leonor = "C:/Users/srale/Downloads/Feature_Selection1_25.csv"
# output_path = "C:/Users/srale/Desktop"
# folder_name = "features_leonor"
# df = pd.read_csv(path_leonor, delimiter=';')
# df['Class'] = df['Class'].astype('int64')
# _,_ = feature_engineering.feature_selector(df, 0.01, 15, "gmm", output_path, folder_name, False)
#
#
