import feature_engineering
import processing
import synchronization
import clustering
from processing.pre_processing import _apply_filters
import load


# #
# path ="D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone/raw_tasks_P020/walking/P020_synchronized_phone_watch_walking_2024-07-15_10_50_26_crosscorr_slow.csv"
# # load data to csv
# data = load.load_data_from_csv(path)
# # filter signals
# filtered_data = _apply_filters(data, 100)
#
# # cut first 200 samples to remove impulse response from the butterworth filters
# filtered_data = filtered_data.iloc[200:]
#
# path_filt = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone/filtered_tasks_P020/walking/P020_synchronized_phone_watch_walking_2024-07-15_10_50_26_crosscorr_slow.csv"
#
# filtered_data.to_csv(path_filt)


def main():
    # Set these booleans to True or False depending on which steps to run

    do_synchronization = False
    do_processing = False
    do_generate_cfg_file = False
    do_feature_extraction = False
    # feature selection
    do_one_subject_feature_selection = False
    do_general_model_feature_selection = False
    do_all_subjects_feature_selection = False
    # clustering
    do_general_model_clustering = True
    do_subject_specific_clustering = False

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P020"
        sync_android_out_path = ("D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone/sync_android_P020")
        output_path = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone/synchronized_P020"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}  #
        sync_type = 'crosscorr'
        evaluation_path = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone"
        synchronization.synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path,
                                        sync_type, evaluation_path)

    if do_processing:
        output_path = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone"
        filtered_folder_name = "filtered_tasks_P020"
        raw_folder_name = "raw_tasks_P020"
        sync_data_path = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch_phone/synchronized_P020"
        processing.processor(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        feature_engineering.generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium',
                      'sit']  # , 'standing_gestures', 'stairs', 'walk_fast', 'walk_slow', 'coffee', 'folders'
        main_path = "D:/tese_backups/subjects/P020/acc_gyr_mag_watch/filtered_tasks_P020"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P020"
        output_filename = "acc_gyr_mag_watch_features_P020.csv"
        output_folder_name = "watch_features_basic_activities"
        feature_engineering.feature_extractor(main_path, output_path, subclasses, output_filename, output_folder_name)

    if do_one_subject_feature_selection:
        # #
        # import os
        # main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        # features_folder_name = "watch_features_basic_activities"
        # # iterate through the subject folders
        # for subject_folder in os.listdir(main_path):
        #     subject_folder_path = os.path.join(main_path, subject_folder)
        #
        #     # iterate through the folders inside each subject folder
        #     for folder_name in os.listdir(subject_folder_path):
        #
        #         # get the specified folder
        #         if folder_name == features_folder_name:
        #
        #             # get the path to the dataset
        #             features_folder_path = os.path.join(subject_folder_path, features_folder_name)
        #
        #             # check if there's only one csv file in the folder
        #             if len(os.listdir(features_folder_path)) == 1:
        #                 # only one csv file for the features folder
        #                 dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])
        #
        #                 output_path_plots = "D:/tese_backups/subjects/P004/acc_gyr_mag_phone"
        #                 # # train test split
        #                 # train_set, _ = load.train_test_split(dataset_path, 0.8, 0.2)
        #                 #
        #                 #
        #                 # _, _, _, _ = feature_engineering.feature_selector(train_set, 0.05, 0.99, 20,
        #                 #                                                   "kmeans", output_path_plots)
        #                 results_output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/results_gmm"
        #                 results_folder_name = "subject_specific_basic_feature_selection"
        #                 results_filename_prefix = "watch_basic_80_20"
        #                 feature_engineering.one_stage_feature_selection(dataset_path, 0.05, 0.99,
        #                                                                 10, "gmm", output_path_plots,
        #                                                                 results_output_path,
        #                                                                 results_folder_name, results_filename_prefix)

        dataset_path = (
            "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P019/phone_features_basic_activities"
            "/acc_gyr_mag_phone_features_P019.csv")
        output_path_plots = "D:/tese_backups/subjects/P010/acc_gyr_mag_phone"
        # # train test split
        # train_set, _ = load.train_test_split(dataset_path, 0.8, 0.2)
        #
        #
        # _, _, _, _ = feature_engineering.feature_selector(train_set, 0.05, 0.99, 20,
        #                                                   "kmeans", output_path_plots)
        results_output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/results_gmm"
        results_folder_name = "subject_specific_basic_feature_selection"
        results_filename_prefix = "phone_basic_80_20"
        feature_engineering.one_stage_feature_selection(dataset_path, 0.05, 0.99,
                                                        13, "gmm", output_path_plots, results_output_path,
                                                        results_folder_name, results_filename_prefix)

    if do_general_model_feature_selection:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        subfolder_name = "watch_phone_features_basic_activities"
        output_path = "D:/tese_backups/general_model"

        all_train_sets = load.load_all_subjects(main_path, subfolder_name, False)
        _, _, _ = feature_engineering.feature_selector(all_train_sets, 0.01, 0.99, 10, "gmm", output_path)

    if do_all_subjects_feature_selection:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        features_folder_name = "watch_phone_features_all_activities"
        clustering_model = "kmeans"
        feature_selection_iterations = 6
        variance_threshold = 0.05
        correlation_threshold = 0.99
        top_1n = 10  # Number of top features to select
        nr_iterations = 4

        (best_feature_set_with_axis16, ari_with_axis16, nmi_with_axis16,  best_feature_set_without_axis16, ari_without_axis16,
         nmi_without_axis16) = feature_engineering.two_stage_feature_selection(main_path, features_folder_name, variance_threshold,
                                                        correlation_threshold,
                                                        feature_selection_iterations, clustering_model, nr_iterations,
                                                        top_1n)

        # # ####
        # top_2n = 10
        # (best_feature_set_with_axis18, ari_with_axis18, nmi_with_axis18,  best_feature_set_without_axis18, ari_without_axis18,
        #  nmi_without_axis18) = feature_engineering.two_stage_feature_selection(main_path, features_folder_name,
        #                                                                        variance_threshold,
        #                                                                        correlation_threshold,
        #                                                                        feature_selection_iterations,
        #                                                                        clustering_model, nr_iterations,
        #                                                                        top_2n)
        #
        # top_n3 = 12
        # (best_feature_set_with_axis6, ari_with_axis6, nmi_with_axis6, best_feature_set_without_axis6,
        #  ari_without_axis6,
        #  nmi_without_axis6) = feature_engineering.two_stage_feature_selection(main_path, features_folder_name,
        #                                                                        variance_threshold,
        #                                                                        correlation_threshold,
        #                                                                        feature_selection_iterations,
        #                                                                        clustering_model, nr_iterations,
        #                                                                        top_n3)
        # top_n4 = 14
        # (best_feature_set_with_axisx, ari_with_axisx, nmi_with_axisx, best_feature_set_without_axisx,
        #  ari_without_axisx,
        #  nmi_without_axisx) = feature_engineering.two_stage_feature_selection(main_path, features_folder_name,
        #                                                                       variance_threshold,
        #                                                                       correlation_threshold,
        #                                                                       feature_selection_iterations,
        #                                                                       clustering_model, nr_iterations,
        #                                                                       top_n4)
        #
        # top_n5 = 16
        # (best_feature_set_with_axisy, ari_with_axisy, nmi_with_axisy, best_feature_set_without_axisy,
        #  ari_without_axisy,
        #  nmi_without_axisy) = feature_engineering.two_stage_feature_selection(main_path, features_folder_name,
        #                                                                       variance_threshold,
        #                                                                       correlation_threshold,
        #                                                                       feature_selection_iterations,
        #                                                                       clustering_model, nr_iterations,
        #                                                                       top_n5)
        #
        #
        #
        #
        #
        # print("RESULTS IN ORDER")
        # print(top_1n)
        # print(f"best feature set with axis {best_feature_set_with_axis16}")
        # print(f"ARI {ari_with_axis16}")
        # print(f"NMI {nmi_with_axis16}")
        # print(f"best feature set with axis {best_feature_set_without_axis16}")
        # print(f"ARI {ari_without_axis16}")
        # print(f"NMI {nmi_without_axis16}\n")
        #
        # print(top_2n)
        # print(f"best feature set with axis {best_feature_set_with_axis18}")
        # print(f"ARI {ari_with_axis18}")
        # print(f"NMI {nmi_with_axis18}")
        # print(f"best feature set with axis {best_feature_set_without_axis18}")
        # print(f"ARI {ari_without_axis18}")
        # print(f"NMI {nmi_without_axis18}\n")
        #
        # print(top_n3)
        # print(f"best feature set with axis {best_feature_set_with_axis6}")
        # print(f"ARI {ari_with_axis6}")
        # print(f"NMI {nmi_with_axis6}")
        # print(f"best feature set with axis {best_feature_set_without_axis6}")
        # print(f"ARI {ari_without_axis6}")
        # print(f"NMI {nmi_without_axis6}\n")
        #
        # print(top_n4)
        # print(f"best feature set with axis {best_feature_set_with_axisx}")
        # print(f"ARI {ari_with_axisx}")
        # print(f"NMI {nmi_with_axisx}")
        # print(f"best feature set with axis {best_feature_set_without_axisx}")
        # print(f"ARI {ari_without_axisx}")
        # print(f"NMI {nmi_without_axisx}\n")
        #
        # print(top_n5)
        # print(f"best feature set with axis {best_feature_set_with_axisy}")
        # print(f"ARI {ari_with_axisy}")
        # print(f"NMI {nmi_with_axisy}")
        # print(f"best feature set with axis {best_feature_set_without_axisy}")
        # print(f"ARI {ari_without_axisy}")
        # print(f"NMI {nmi_without_axisy}\n")


    if do_general_model_clustering:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        subfolder_name = "phone_features_all_activities"
        clustering_model = "kmeans"
        results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
        feature_set = ['zMag_Mean', 'zAcc_Median frequency', 'xMag_Mean', 'xGyr_Spectral centroid', 'yGyr_Standard deviation', 'zGyr_Standard deviation', 'xAcc_Max', 'zGyr_Spectral entropy', 'yAcc_Spectral entropy', 'xGyr_Standard deviation', 'xGyr_Min', 'yGyr_Spectral entropy', 'xAcc_Median frequency', 'xMag_Min', 'xMag_Spectral centroid']

        _, _ = clustering.general_model_clustering(main_path, subfolder_name, clustering_model, feature_set, results_path)

    if do_subject_specific_clustering:
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        features_folder_name = "phone_features_basic_activities"
        clustering_model = "gmm"
        feature_sets_path = ("C:/Users/srale/OneDrive - FCT NOVA/Tese/results_gmm/subject_specific_basic_feature_selection"
                             "/phone_basic_80_20_feature_selection_results.txt")
        results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
        clustering.subject_specific_clustering(main_path, feature_sets_path, clustering_model, features_folder_name, results_path)


if __name__ == "__main__":
    main()

# from feature_engineering.test_on_all_subjects import _test_same_feature_set_for_all_subjects
#
# # test two stage
# main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
# features_folder_name = "watch_features_all_activities"
# feature_set =  ['xMag_wear_Max', 'xGyr_wear_Interquartile range', 'xAcc_wear_Median frequency', 'xAcc_wear_Spectral entropy', 'xGyr_wear_Spectral entropy', 'xMag_wear_Spectral entropy', 'xMag_wear_Min', 'xMag_wear_Mean', 'xAcc_wear_Spectral centroid', 'xAcc_wear_Standard deviation', 'xGyr_wear_Standard deviation', 'xMag_wear_Median frequency', 'xGyr_wear_Max', 'xGyr_wear_Median frequency', 'xMag_wear_Standard deviation', 'xAcc_wear_Max']
#
#
# _test_same_feature_set_for_all_subjects(main_path, features_folder_name, "kmeans", feature_set)



#
#
#
#
#

#
#
#
#
#



