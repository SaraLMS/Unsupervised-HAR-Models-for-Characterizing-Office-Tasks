import feature_engineering
import processing
import synchronization
import clustering

do_synchronization = False
do_processing = False
do_feature_extraction = False
# feature selection
run_experiment1 = False
# experiment 2
run_experiment2 = False
# experiment 3
run_experiment3 = True


def main():
    # Set these booleans to True or False depending on which steps to run

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions"
        output_path = "D:/tese_backups/test_new_synchronization"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        synchronization.synchronize_all(raw_data_in_path, selected_sensors, sync_type, output_path)

    if do_processing:
        output_path = "D:/tese_backups/test_new_synchronization"
        sync_data_path = "D:/tese_backups/test_new_synchronization"
        devices_sensors_foldername = "acc_gyr_mag_phone_watch"
        processing.process_all(sync_data_path, output_path, devices_sensors_foldername)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']  # , 'standing_gestures', 'stairs', 'walk_fast', 'walk_slow', 'coffee', 'folders'
        main_path = "D:/tese_backups/test_new_synchronization"
        output_path = "D:/tese_backups/test_new_synchronization/datasets"
        devices_folder_name = "acc_gyr_mag_phone_watch"
        feature_engineering.feature_extraction_all(main_path, devices_folder_name, output_path, subclasses)

    if run_experiment1:
        with open('run_experiment1.py') as f:
            code = f.read()
            exec(code)

    if run_experiment2:
        with open('run_experiment2.py') as f:
            code = f.read()
            exec(code)

    if run_experiment3:
        with open('run_experiment3.py') as f:
            code = f.read()
            exec(code)


if __name__ == "__main__":
    main()

















# from feature_engineering.test_on_all_subjects import _test_same_feature_set_for_all_subjects
#
# # test two stage
# main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
# features_folder_name = "watch_phone_features_all_activities"
# feature_set =  ['yAcc_Standard deviation', 'xMag_wear_Max', 'xAcc_Standard deviation', 'yGyr_Standard deviation', 'zGyr_wear_Standard deviation', 'xGyr_Min', 'yMag_wear_Spectral centroid', 'yAcc_Max']
#
# _test_same_feature_set_for_all_subjects(main_path, features_folder_name, "kmeans", feature_set)
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