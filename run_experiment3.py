# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
import clustering

# ------------------------------------------------------------------------------------------------------------------- #
# Experiment 3
# ------------------------------------------------------------------------------------------------------------------- #


main_path = "D:/tese_backups/subjects_datasets"
features_folder_name = "phone_features_basic_activities"
clustering_model = "kmeans"
results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
feature_set = ['xMag_Max', 'yAcc_Interquartile range', 'zMag_Max', 'xAcc_Min', 'yAcc_Min']

sitting_perc = 0.9
nr_chunks = 20

clustering.unbalanced_clustering(main_path, sitting_perc, nr_chunks, clustering_model, 3, features_folder_name,
                                 feature_set, results_path)