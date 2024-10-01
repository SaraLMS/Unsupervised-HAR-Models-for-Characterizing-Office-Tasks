# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

# internal imports
import clustering

# ------------------------------------------------------------------------------------------------------------------- #
# Experiment 3
# ------------------------------------------------------------------------------------------------------------------- #


main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects_datasets"
features_folder_name = "phone_features_basic_activities"
clustering_model = "agglomerative"
results_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
feature_set = ['yAcc_Max', 'zAcc_Interquartile range', 'zMag_Max', 'yMag_Max']

sitting_perc = 0.9
nr_chunks = 20

clustering.unbalanced_clustering(main_path, sitting_perc, nr_chunks, clustering_model, 3, features_folder_name,
                                 feature_set, results_path)