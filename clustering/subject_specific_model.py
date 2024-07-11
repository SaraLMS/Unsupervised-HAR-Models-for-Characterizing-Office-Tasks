# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import ast
from typing import List

import pandas as pd
import numpy as np
import os

# internal imports
import load
import metrics
from constants import CLASS, SUBJECT_ID, FEATURE_SET
from .common import normalize_features, cluster_data


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def subject_specific_clustering(main_path: str, subjects_features_path: str, clustering_model: str, features_folder_name: str):

    # load the csv file with the subject id's and the respective feature sets
    # columns are the subject id's and the feature set to be used for that subject
    feature_sets = pd.read_csv(subjects_features_path, delimiter=";")

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

                    # find the subject id in the subject features df to get the feature set for that subject
                    if subject_folder in feature_sets[SUBJECT_ID].values:

                        subject_feature_set_str = \
                            feature_sets.loc[feature_sets[SUBJECT_ID] == subject_folder, FEATURE_SET].values[0]

                        # Convert feature_set string into a list of strings
                        subject_feature_set = _parse_feature_set(subject_feature_set_str)

                        if subject_feature_set:
                            # Cluster the subject
                            ri, ari, nmi = cluster_subject(dataset_path, clustering_model, subject_feature_set)

                            # Inform user
                            print(f"Clustering results for subject: {subject_folder}")
                            print(f"Feature set used: {subject_feature_set_str}")
                            print(
                                f"Rand Index: {ri}; Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")
                        else:
                            print(f"Failed to parse feature set for subject {subject_folder}. Skipping.")
                    else:
                        raise ValueError(f"Subject ID {subject_folder} not found in the feature sets CSV.")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")


def cluster_subject(dataset_path:str, clustering_model: str, feature_set: List[str], train_size: float = 0.9,
                    test_size: float = 0.1):
    # train test split
    train_set, test_set = load.train_test_split(dataset_path, train_size, test_size)

    # Check if all features in the feature set exist in the dataframe columns
    _check_features(train_set, feature_set)
    _check_features(test_set, feature_set)

    # get true labels for evaluation
    true_labels = test_set[CLASS]

    # get only the wanted features in the train and test sets
    train_set = train_set[feature_set]
    test_set = test_set[feature_set]

    # normalize the features
    train_set = normalize_features(train_set)
    test_set = normalize_features(test_set)

    # cluster data
    pred_labels = cluster_data(clustering_model, train_set, test_set, n_clusters=3)

    # evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    return ri, ari, nmi


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
def _parse_feature_set(feature_set_str):
    """
    Parses the feature set string into a list of features.

    :param feature_set_str: str
    A string representation of a list of features.

    :return: list
    A list of features.
    """
    try:
        return ast.literal_eval(feature_set_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing feature set: {e}")
        return []


def _check_features(dataframe, feature_set):
    missing_features = [feature for feature in feature_set if feature not in dataframe.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the dataframe columns: {missing_features}")
