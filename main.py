from feature_engineering.feature_selection import feature_selector, get_top_features_across_all_subjects, \
    get_all_subjects_best_features, test_feature_set_each_subject, test_different_axis
from feature_extraction.feature_extraction import feature_extractor, generate_cfg_file
from processing.processor import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals


def main():
    # Set these booleans to True or False depending on which steps to run
    do_synchronization = False
    do_processing = False
    do_generate_cfg_file = False
    do_feature_extraction = False
    do_one_subject_feature_selection = False
    do_all_subjects_feature_selection = True

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P010"
        sync_android_out_path = ("C:/Users/srale/OneDrive - FCT "
                                 "NOVA/Tese/subjects/P010/acc_gyr_mag_phone/sync_android_P010")
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone/synchronized_P010"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone/sync_evaluation_P010"
        synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,
                        evaluation_path)

    if do_processing:
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone"
        filtered_folder_name = "filtered_tasks_P010"
        raw_folder_name = "raw_tasks_P010"
        sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone/sync_android_P010"
        processing(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P010"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        feature_extractor(main_path, output_path, subclasses)

    if do_one_subject_feature_selection:
        dataset_path = (
            "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone/features_basic_activities"
            "/acc_gyr_mag_phone_features_P010.csv")
        output_path_plots = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P010/acc_gyr_mag_phone"
        feature_sets, best_acc = feature_selector(dataset_path, 0.05, 20, "kmeans", output_path_plots)
        print(feature_sets)

    if do_all_subjects_feature_selection:
        subject_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        features_folder_name = "features_basic_activities"
        clustering_model = "kmeans"
        nr_iterations = 10

        top_n = 8  # Number of top features to select

        # subjects_dict = get_all_subjects_best_features(subject_path, features_folder_name, 0.05, nr_iterations,
        #                                                clustering_model)
        #
        # final_feature_set = get_top_features_across_all_subjects(subjects_dict, 6)
        #
        # mean_ri, mean_ari, mean_nmi = test_feature_set_each_subject(subject_path, features_folder_name,
        #                                                             clustering_model, final_feature_set)
        #
        # print(f"Mean rand index: {mean_ri} \nMean adjusted rand index: {mean_ari}\n"
        #       f"Mean normalized mutual information: {mean_nmi}")
        # Get the best features for each subject
        subjects_dict = get_all_subjects_best_features(subject_path, features_folder_name, 0.05, nr_iterations,
                                                       clustering_model)

        # Get the top features with axis
        final_feature_set_with_axis = get_top_features_across_all_subjects(subjects_dict, top_n)['features_with_axis']

        # Test the top features with axis for each subject
        mean_ri, mean_ari, mean_nmi = test_feature_set_each_subject(subject_path, features_folder_name,
                                                                    clustering_model, final_feature_set_with_axis)

        print(f"Results for features with axis:")
        print(f"Mean Rand Index: {mean_ri}")
        print(f"Mean Adjusted Rand Index: {mean_ari}")
        print(f"Mean Normalized Mutual Information: {mean_nmi}")

        # Test the top features without axis by adding different axes
        axis_test_results = test_different_axis(subjects_dict, subject_path, features_folder_name, clustering_model,
                                                top_n)

        print("Results for testing different axes:")
        for axis, results in axis_test_results.items():
            print(
                f"Axis: {axis} -> Mean RI: {results['mean_ri']}, Mean ARI: {results['mean_ari']}, Mean NMI: {results['mean_nmi']}")


if __name__ == "__main__":
    main()
