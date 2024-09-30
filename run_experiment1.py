# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
import feature_engineering
import load

# ------------------------------------------------------------------------------------------------------------------- #
# EXPERIMENT 1
# ------------------------------------------------------------------------------------------------------------------- #

# Set these booleans to True or False depending on which models to run
run_exp1_SSM = False
run_exp1_1GM = False
run_exp1_2GM = False

# ------------------------------------------------------- SSM ------------------------------------------------------- #
if run_exp1_SSM:
    dataset_path = (
        "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets/P019/phone_features_basic_activities"
        "/acc_gyr_mag_phone_features_P019.csv")
    output_path_plots = "D:/tese_backups/subjects/P010/acc_gyr_mag_phone"
    results_output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/RESULTS/gmm"
    results_folder_name = "subject_specific_basic_feature_selection"
    results_filename_prefix = "phone_basic_80_20"
    feature_engineering.one_stage_feature_selection(dataset_path, 0.05, 0.99,
                                                    10, "gmm", output_path_plots,
                                                    results_output_path, results_folder_name, results_filename_prefix)

# ------------------------------------------------------- 1GM ------------------------------------------------------- #
if run_exp1_1GM:
    main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
    subfolder_name = "watch_features_all_activities"
    output_path = "D:/tese_backups/general_model"

    all_train_sets, _ = load.load_all_subjects(main_path, subfolder_name)
    _, _, _ = feature_engineering.feature_selector(all_train_sets, 0.01, 0.99,
                                                   5, "gmm", output_path)

# ------------------------------------------------------- 2GM ------------------------------------------------------- #
if run_exp1_2GM:
    main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
    features_folder_name = "watch_phone_features_all_activities"
    clustering_model = "kmeans"
    feature_selection_iterations = 4
    variance_threshold = 0.05
    correlation_threshold = 0.99
    top_1n = 8  # Number of top features to select
    nr_iterations = 4

    _, _, _, _, _, _ = feature_engineering.two_stage_feature_selection(main_path, features_folder_name,
                                                                       variance_threshold,
                                                                       correlation_threshold,
                                                                       feature_selection_iterations,
                                                                       clustering_model, nr_iterations,
                                                                       top_1n)
