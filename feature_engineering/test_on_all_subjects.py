from .feature_selection import  _test_feature_set
from typing import List, Tuple
import os
import numpy as np
import pandas as pd



def _test_same_feature_set_for_all_subjects(main_path: str, features_folder_name: str, clustering_model: str,
                                            feature_set: List[str]) -> Tuple[float, float]:
    """
    Performs clustering and evaluates the performance across all subjects with the same feature set.
    This function goes through the main folder containing the subject folders. Inside each subject folder finds
    the folder with the name features_folder_name where the csv file containing all the features is stored. Loads this
    file, gets only the columns with the chose features is feature_set, clusters the data and evaluates. Returns the
    mean of the rand index, adjusted rand index and normalized mutual information of all subjects.

    :param main_path: str
    Path to the main folder containing the subject folders

    :param features_folder_name: str
    Name of the folder containing the csv file with the features

    :param clustering_model: str
    Unsupervised learning model used to test the feature set. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model
        "dbscan": DBSCAN. Needs parameter search - not implemented
        "birch": Birch clustering algorithm

    :param feature_set: List[str]
    List of features to be used for clustering

    :return: Tuple[float, float, float]
    Mean scores of all subjects
    """
    results = []
    ari_list = []
    nmi_list = []

    # iterate through the folders of each subject
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)

        # iterate through the sub folders
        for sub_folder in os.listdir(subject_folder_path):

            if sub_folder == features_folder_name:

                features_folder_path = os.path.join(subject_folder_path, sub_folder)

                # list with the path to the csv file
                feature_files = os.listdir(features_folder_path)

                if len(feature_files) == 1:
                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, feature_files[0])

                    ari, nmi = _test_feature_set(feature_set, dataset_path, clustering_model)
                    results.append({"Subject": subject_folder, "ARI": ari, "NMI": nmi})

                    # Append the results to the lists

                    ari_list.append(ari)
                    nmi_list.append(nmi)

                else:
                    raise ValueError(f"Too many files: {len(feature_files)}")

        # Create a DataFrame from the results
        df = pd.DataFrame(results)

        excel_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels"
        # Save to Excel
        excel_path = os.path.join(excel_path, "agg_basic_phone6.xlsx")
        df.to_excel(excel_path, index=False)

    print(
        f"Avg Adjusted Rand Index: {np.round(np.mean(ari_list), 4)}\n"
        f"Avg Normalized Mutual Information: {np.round(np.mean(nmi_list), 4)}")

    return np.round(np.mean(ari_list), 4), np.round(np.mean(nmi_list), 4)




