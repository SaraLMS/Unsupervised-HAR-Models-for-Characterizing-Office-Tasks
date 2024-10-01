# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

import load
from constants import CLASS, SUBCLASS, SUPPORTED_MODELS
from .common import cluster_subject_all_activities, cluster_subject_basic_matrix
from typing import List
import os

from .random_forest import random_forest_classifier


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def two_stage_general_model_clustering(main_path: str, clustering_model: str, nr_clusters: int, features_folder_name: str,
                                       feature_set: List[str], results_path: str, plots_path: str, activities: str,
                                       save_results_in_excel_sheet: bool = True) -> None:
    """
    This function implements experiment 2 for the two-stage general model. For more information check README file.
    In this experiment, clustering is performed on the train set and the test set instances are appointed to the
    preformed clusters. If Agglomerative Clustering is chosen, clustering is performed on 20 % of the data.
    This function also trains a random forest with the same feature sets used for the clustering.

    This function generates and saves confusion matrices and saves the adjusted rand index, normalized mutual
    information, and accuracy score (random forest) results in an Excel sheet.

    :param main_path: str
    Path to the main_folder containing subfolders which have the datasets. The directory scheme is the following
    main_folder/folder/subfolder/dataset.csv
    (i.e., datasets/features_basic_acc_gyr_mag_phone_watch/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param features_folder_name: str
    Path to the folder[*] identifying which datasets to load. The directory scheme is the following
    folder[*]/subfolder/dataset.csv
    (i.e., features_basic_acc_gyr_mag_phone_watch[*]/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :param results_path: str
    Path to the directory where to save the Excel sheet with the clustering and random forest results.

    :param plots_path: str
    Path to the directory where to save the confusion matrices

    :param activities: str
    Activities being clustered. Suppored activities are:
        'basic': 'sitting', 'standing_still', 'walking_medium'
        'all': 'sitting', 'standing_still', 'walking_medium', 'standing_gestures', 'stairs',
                'walk_fast', 'walk_slow', 'coffee', 'folders'

    :param save_results_in_excel_sheet: bool (default = True)
    Save the results in an Excel sheet in the results_path directory. False not to save

    :return: None
    """
    # check supported models
    _check_supported_models(clustering_model)

    # list for holding the results on each subject
    results = []

    # iterate through the subject folders
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)

        # iterate through the folders inside each subject folder
        for folder_name in os.listdir(subject_folder_path):

            # get the specified folder with the devices and sensors chosen
            if folder_name == features_folder_name:

                # get the path to the dataset
                features_folder_path = os.path.join(subject_folder_path, features_folder_name)

                # check if there's only one csv file in the folder
                if len(os.listdir(features_folder_path)) == 1:
                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])

                    if activities == 'basic':
                        ari, nmi = cluster_subject_basic_matrix(dataset_path, clustering_model, feature_set,
                                                                subject_folder, plots_path)

                    elif activities == 'all':
                        ari, nmi = cluster_subject_all_activities(dataset_path, clustering_model, nr_clusters, feature_set,
                                                                  subject_folder, plots_path)

                    else:
                        raise ValueError(f"The activities {activities} are not supported. Only 'basic' and 'all'.")

                    # for the random forest
                    # (1) train test split
                    train_set, test_set = load.train_test_split(dataset_path, 0.8, 0.2)

                    # x, y split
                    x_train = train_set.drop([CLASS, SUBCLASS], axis=1)
                    y_train = train_set[CLASS]
                    x_test = test_set.drop([CLASS, SUBCLASS], axis=1)
                    y_test = test_set[CLASS]

                    # (2) train a random forest
                    accuracy_score = random_forest_classifier(x_train, x_test, y_train, y_test)

                    # save the results in a dictionary
                    results.append({
                        "Subject ID": subject_folder,
                        "ARI": ari,
                        "NMI": nmi,
                        "Random Forest acc": accuracy_score
                    })
                    # Inform user
                    print(f"Clustering results for subject: {subject_folder}")

                    # inform user of the clustering results for each subject/folder inside main_path
                    print(
                        f"Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")

                # each folder must only have one csv file inside
                else:
                    raise ValueError("Only one dataset per folder is allowed.")

    # if true, save clustering and random forest results to an Excel sheet
    if save_results_in_excel_sheet:

        # Create DataFrame from results and save to Excel
        results_df = pd.DataFrame(results)

        # filename with clustering model, name of the folder pertaining to the devices/sensors and activities
        results_excel_filename = f"clustering_results_{features_folder_name}_{clustering_model}_{activities}_activities.xlsx"

        # generate path
        excel_path = os.path.join(results_path, results_excel_filename)

        # save Excel sheet
        results_df.to_excel(excel_path, index=False)

        # inform user
        print(f"Results saved to {excel_path}")

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _check_supported_models(clustering_model: str):
    """
    Check is the chosen model is in the supported sensors. If not, raises a value error.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :return: None
    """
    if clustering_model not in SUPPORTED_MODELS:
        raise ValueError(f"{clustering_model} is no supported. Supported models are: {SUPPORTED_MODELS}")











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
