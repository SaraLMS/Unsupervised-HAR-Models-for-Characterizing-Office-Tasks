# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
import clustering

# ------------------------------------------------------------------------------------------------------------------- #
# Experiment 2
# ------------------------------------------------------------------------------------------------------------------- #

# Set these booleans to True or False depending on which models to run
run_exp2_SSM = False
run_exp2_1GM = False
run_exp2_2GM = False

# ------------------------------------------------------- SSM ------------------------------------------------------- #

if run_exp2_SSM:
    main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
    features_folder_name = "watch_phone_features_all_activities"
    clustering_model = "gmm"
    feature_sets_path = (
        "D:/tese_backups/RESULTS/gmm/subject_specific_all_feature_selection"
        "/watch_phone_all_80_20_feature_selection_results.txt")
    results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
    clustering.subject_specific_clustering(main_path, feature_sets_path, clustering_model, 3, features_folder_name,
                                           results_path)

# ------------------------------------------------------- 1GM ------------------------------------------------------- #
if run_exp2_1GM:
    main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
    subfolder_name = "phone_features_all_activities"
    clustering_model = "agglomerative"
    results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
    feature_set = ['xGyr_Standard deviation', 'yAcc_Interquartile range', 'zMag_Max', 'xGyr_Spectral entropy',
                   'yGyr_Min', 'xMag_Spectral centroid']

    clustering.one_stage_general_model_each_subject(main_path, subfolder_name, clustering_model, 3, feature_set,
                                                    results_path)

# ------------------------------------------------------- 2GM ------------------------------------------------------- #
if run_exp2_2GM:
    main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
    features_folder_name = "phone_features_basic_activities"
    results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
    confusion_matrix_path = "D:/tese_backups/RESULTS/kmeans/clustering/confusion_matrix_2_stage/basic_phone"
    feature_set = ['xMag_Max', 'yAcc_Interquartile range', 'zMag_Max', 'xAcc_Min', 'yAcc_Min']

    clustering.two_stage_general_model_clustering(main_path, "kmeans", 3, features_folder_name,
                                                  feature_set, results_path, confusion_matrix_path, 'basic')
