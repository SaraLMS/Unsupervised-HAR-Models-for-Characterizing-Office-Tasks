# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

import load
from constants import CLASS, SUBCLASS
from .common import cluster_subject_all_activities, cluster_subject_basic_matrix
from typing import List
import os

from .random_forest import random_forest_classifier


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def two_stage_general_model_clustering(main_path: str, clustering_model: str, features_folder_name: str,
                                       feature_set: List[str], results_path: str, plots_path: str):
    # list for holding the results on each subject
    results = []

    # iterate through the subject folders
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)

        # iterate through the folders inside each subject folder
        for folder_name in os.listdir(subject_folder_path):

            # get the specified folder
            if folder_name == features_folder_name:

                # get the path to the dataset
                features_folder_path = os.path.join(subject_folder_path, features_folder_name)

                # check if there's only one csv file in the folder
                if len(os.listdir(features_folder_path)) == 1:
                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])

                    ari, nmi = cluster_subject_basic_matrix(dataset_path, clustering_model, feature_set, subject_folder, plots_path)

                    # test random forest
                    # train test split
                    train_set, test_set = load.train_test_split(dataset_path, 0.8, 0.2)

                    # x, y split
                    x_train = train_set.drop([CLASS, SUBCLASS], axis=1)
                    y_train = train_set[CLASS]
                    x_test = test_set.drop([CLASS, SUBCLASS], axis=1)
                    y_test = test_set[CLASS]

                    # try random forest
                    accuracy_score = random_forest_classifier(x_train, x_test, y_train, y_test)

                    results.append({
                        "Subject ID": subject_folder,
                        "ARI": ari,
                        "NMI": nmi,
                        "Random Forest acc": accuracy_score
                    })
                    # Inform user
                    print(f"Clustering results for subject: {subject_folder}")
                    # print(f"Feature set used: {subject_feature_set_str}")
                    print(
                        f"Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")

    # Create DataFrame from results and save to Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(results_path, "clustering_results_kmeans_rf_basic_phone.xlsx")
    results_df.to_excel(excel_path, index=False)

    print(f"Results saved to {excel_path}")
















########################### OLD ###########################################

# def two_stage_general_model_unbalanced(main_path: str, clustering_model: str, features_folder_name: str,
#                                        feature_set: List[str], results_path: str):
#     # list for holding the results on each subject
#     results = []
#
#     # iterate through the subject folders
#     for subject_folder in os.listdir(main_path):
#         subject_folder_path = os.path.join(main_path, subject_folder)
#
#         # iterate through the folders inside each subject folder
#         for folder_name in os.listdir(subject_folder_path):
#
#             # get the specified folder
#             if folder_name == features_folder_name:
#
#                 # get the path to the dataset
#                 features_folder_path = os.path.join(subject_folder_path, features_folder_name)
#
#                 # check if there's only one csv file in the folder
#                 if len(os.listdir(features_folder_path)) == 1:
#                     # only one csv file for the features folder
#                     dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])
#
#                     ari, nmi = cluster_unbalanced_data(dataset_path, clustering_model, feature_set)
#
#                     results.append({
#                         "Subject ID": subject_folder,
#                         "ARI": ari,
#                         "NMI": nmi,
#                     })
#                     # Inform user
#                     print(f"Clustering results for subject: {subject_folder}")
#                     # print(f"Feature set used: {subject_feature_set_str}")
#                     print(
#                         f"Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")
#
#                 else:
#                     raise ValueError("Only one dataset per folder is allowed.")
#
#     # Create DataFrame from results and save to Excel
#     results_df = pd.DataFrame(results)
#     excel_path = os.path.join(results_path, "clustering_results_gmm_basic_watch_phone.xlsx")
#     results_df.to_excel(excel_path, index=False)
#
#     print(f"Results saved to {excel_path}")
