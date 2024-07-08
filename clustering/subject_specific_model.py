# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import os

# internal imports
import load
import metrics
from .common import normalize_features, cluster_data


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def cluster_all_subjects(main_path: str, subjects_features_path: str, clustering_model: str, features_folder_name: str):
    # lists for saving the scores for all subjects
    ri_list = []
    ari_list = []
    nmi_list = []

    # load the csv file with the subject id's and the respective feature sets
    # columns are the subject id's and the feature set to be used for that subject
    feature_sets = pd.read_csv(subjects_features_path)

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

                    # load dataset
                    df = load.load_data_from_csv(dataset_path)

                    # find the subject id in the subject features df to get the feature set for that subject
                    if subject_folder in feature_sets['subject_id'].values:
                        subject_feature_set = \
                        feature_sets.loc[feature_sets['subject_id'] == subject_folder, 'feature_set'].values[0]

                        # cluster the subject
                        ri, ari, nmi = cluster_subject(df, clustering_model, subject_feature_set)

                        # add to the respective lists
                        ri_list.append(ri)
                        ari_list.append(ari)
                        nmi_list.append(nmi)

                        # inform user
                        print(f"Clustering results for subject: {subject_folder}")
                        print(f"Rand Index: {ri}; Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")
                    else:
                        print(f"Subject ID {subject_folder} not found in the feature sets CSV.")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")
            else:
                raise ValueError(f"Folder: {features_folder_name} not found.")

    return np.round(np.mean(ri_list), 2), np.round(np.mean(ari_list), 2), np.round(np.mean(nmi_list), 2)


def cluster_subject(df: pd.DataFrame, clustering_model: str, feature_set: str, train_size: float = 0.7,
                    test_size: float = 0.3):
    # Check if all features in the feature set exist in the dataframe columns
    missing_features = [feature for feature in feature_set if feature not in df.columns]
    if missing_features:
        raise ValueError(f"The following features are not in the dataset columns: {missing_features}")

    # normalize the features
    df = normalize_features(df)

    # train test split
    train_set, test_set = load.train_test_split(df, train_size, test_size)

    # get true labels for evaluation
    true_labels = test_set['class']

    # get only the wanted features in the train and test sets
    train_set = train_set[feature_set]
    test_set = test_set[feature_set]

    # cluster data
    pred_labels = cluster_data(clustering_model, train_set, test_set, n_clusters=3)

    # evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    return ri, ari, nmi
