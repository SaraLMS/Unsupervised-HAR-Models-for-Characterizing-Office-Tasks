# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import numpy as np

from .common import cluster_subject
from typing import List, Tuple
import os


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def subject_specific_clustering(main_path: str, clustering_model: str, features_folder_name: str, feature_set: List[str]):

    # lists to store the metrics
    list_ri = []
    list_ari = []
    list_nmi = []

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

                    ri, ari, nmi = cluster_subject(dataset_path, clustering_model, feature_set)
                    # Inform user
                    print(f"Clustering results for subject: {subject_folder}")
                    # print(f"Feature set used: {subject_feature_set_str}")
                    print(
                        f"Rand Index: {ri}; Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")

    mean_ri = np.round(np.mean(list_ri), 2)
    mean_ari = np.round(np.mean(list_ari), 2)
    mean_nmi = np.round(np.mean(list_nmi), 2)

    print(
          f"Avg Rand Index: {mean_ri}\n"
          f"Avg Adjusted Rand Index: {mean_ari}\n"
          f"Avg Normalized Mutual Information: {mean_nmi}")
