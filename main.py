import feature_engineering
import processing
import synchronization
from processing.pre_processing import _apply_filters
import load


#
# path ="D:/tese_backups/subjects/P012/acc_gyr_mag_watch/raw_tasks_P012/walking/P012_synchronized_phone_watch_walking_2024-06-25_14_30_59_crosscorr_slow.csv"
# # load data to csv
# data = load.load_data_from_csv(path)
# # filter signals
# filtered_data = _apply_filters(data, 100)
#
# # cut first 200 samples to remove impulse response from the butterworth filters
# filtered_data = filtered_data.iloc[200:]
#
# path_filt = "D:/tese_backups/subjects/P012/acc_gyr_mag_watch/filtered_tasks_P012/walking/P012_synchronized_phone_watch_walking_2024-06-25_14_30_59_crosscorr_slow.csv"
#
# filtered_data.to_csv(path_filt)


def main():
    # Set these booleans to True or False depending on which steps to run
    do_synchronization = False
    do_processing = False
    do_generate_cfg_file = False
    do_feature_extraction = False
    do_one_subject_feature_selection = False
    do_all_subjects_feature_selection = True

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P013"
        sync_android_out_path = ("D:/tese_backups/subjects/P013/acc_gyr_mag_watch/sync_android_P013")
        output_path = "D:/tese_backups/subjects/P013/acc_gyr_mag_watch/synchronized_P013"
        selected_sensors = {'phone': ['acc'], 'watch': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        evaluation_path = "D:/tese_backups/subjects/P013/acc_gyr_mag_watch"
        synchronization.synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path,
                                        sync_type,
                                        evaluation_path)

    if do_processing:
        output_path = "D:/tese_backups/subjects/P013/acc_gyr_mag_watch"
        filtered_folder_name = "filtered_tasks_P013"
        raw_folder_name = "raw_tasks_P013"
        sync_data_path = "D:/tese_backups/subjects/P013/acc_gyr_mag_watch/synchronized_P013"
        processing.processor(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        feature_engineering.generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']
        main_path = "D:/tese_backups/subjects/P011/acc_gyr_mag_phone/filtered_tasks_P011"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P011"
        feature_engineering.feature_extractor(main_path, output_path, subclasses)

    if do_one_subject_feature_selection:
        dataset_path = (
            "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P001/watch_features_basic_activities"
            "/acc_gyr_mag_watch_P001.csv")
        output_path_plots = "D:/tese_backups/subjects/P001/acc_gyr_mag_watch"
        feature_sets, best_acc = feature_engineering.feature_selector(dataset_path, 0.01, 20, "kmeans",
                                                                      output_path_plots)
        print(feature_sets)

    if do_all_subjects_feature_selection:
        subject_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
        features_folder_name = "phone_features_basic_activities"
        clustering_model = "kmeans"
        nr_iterations = 10

        top_n = 5  # Number of top features to select

        best_feature_set_with_axis = None
        best_scores_with_axis = [0, 0, 0]

        best_feature_set_without_axis = None
        best_scores_without_axis = [0, 0, 0]
        best_axis = None

        # Repeat 15 times
        for i in range(15):
            print(
                f"****************************************************************************************************")
            print(
                f"************************************* Iteration {i} **************************************************")

            # Get the best features for each subject
            subjects_dict = feature_engineering.get_all_subjects_best_features(subject_path, features_folder_name, 0.05,
                                                                               nr_iterations, clustering_model)

            # Get the top features with axis
            final_feature_set_with_axis = \
            feature_engineering.get_top_features_across_all_subjects(subjects_dict, top_n)['features_with_axis']

            # Test the top features with axis for each subject
            mean_ri, mean_ari, mean_nmi = feature_engineering.test_feature_set_each_subject(subject_path,
                                                                                            features_folder_name,
                                                                                            clustering_model,
                                                                                            final_feature_set_with_axis)

            # Update the best feature set with axis if the current one is better
            if mean_ri > best_scores_with_axis[0]:
                best_feature_set_with_axis = final_feature_set_with_axis
                best_scores_with_axis = [mean_ri, mean_ari, mean_nmi]

            print(f"Results for features with axis:")
            print(f"Mean Rand Index: {mean_ri}")
            print(f"Mean Adjusted Rand Index: {mean_ari}")
            print(f"Mean Normalized Mutual Information: {mean_nmi}")

            # Test the top features without axis by adding different axes
            axis_test_results = feature_engineering.test_different_axis(subjects_dict, subject_path,
                                                                        features_folder_name, clustering_model, top_n)

            print("Results for testing different axes:")
            for axis, results in axis_test_results.items():
                print(
                    f"Axis: {axis} -> Mean RI: {results['mean_ri']}, Mean ARI: {results['mean_ari']}, Mean NMI: {results['mean_nmi']}")

                # Update the best feature set without axis if the current one is better
                if results['mean_ri'] > best_scores_without_axis[0]:
                    best_feature_set_without_axis = results['features']
                    best_scores_without_axis = [results['mean_ri'], results['mean_ari'], results['mean_nmi']]
                    best_axis = axis

        # Print the best feature set with axis
        print("\nBest feature set with axis:")
        print(f"Features: {best_feature_set_with_axis}")
        print(f"Mean Rand Index: {best_scores_with_axis[0]}")
        print(f"Mean Adjusted Rand Index: {best_scores_with_axis[1]}")
        print(f"Mean Normalized Mutual Information: {best_scores_with_axis[2]}")

        # Print the best feature set without axis for each axis
        print("\nBest feature set without axis:")
        print(f"Axis: {best_axis}")
        print(f"Features: {best_feature_set_without_axis}")
        print(f"Mean Rand Index: {best_scores_without_axis[0]}")
        print(f"Mean Adjusted Rand Index: {best_scores_without_axis[1]}")
        print(f"Mean Normalized Mutual Information: {best_scores_without_axis[2]}")


if __name__ == "__main__":
    main()
