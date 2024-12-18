# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import ast
import pandas as pd
import os

import load
# internal imports
from constants import CLASS, SUBJECT_ID, FEATURE_SET, SUBCLASS
from .common import cluster_subject_all_activities
from .random_forest import random_forest_classifier


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def subject_specific_clustering(main_path: str, subjects_features_path: str, clustering_model: str, nr_clusters: int,
                                features_folder_name: str, results_path: str):
    """
    In this experiment, clustering is performed on the train set and the test set instances are appointed to the
    preformed clusters. If Agglomerative Clustering is chosen, clustering is performed on 20 % of the data.
    This function also trains a random forest with the same feature sets used for the clustering.

    The feature sets used for this model are specific for each subject (or dataset) and are loaded from a txt file
    (subjects_features_path) which has two columns: the first is the subject id, and the second is the best feature
    set found for the respective subject in experiment 1, separated by ';' (i.e., P001; ['xAcc_Mean', 'zMag_Max'])

    This function saves the adjusted rand index, normalized mutual information, and accuracy score (random forest)
    results in an Excel sheet.

    :param main_path: str
    Path to the main_folder containing subfolders which have the datasets. The directory scheme is the following
    main_folder/folder/subfolder/dataset.csv
    (i.e., datasets/features_basic_acc_gyr_mag_phone_watch/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)

    :param subjects_features_path: str
    Path to the txt file containing the features sets for each subject. These should match the same conditions of the
    dataset which will be used for this experiment (i.e., if clustering all activities, the path should be to the
    feature sets found for all activities). Otherwise, this the results obtained will not be correct.


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

    :param results_path: str
    Path to the directory where to save the Excel sheet with the clustering and random forest results.

    :return: None
    """
    # load the csv file with the subject id's and the respective feature sets
    # columns are the subject id's and the feature set to be used for that subject
    feature_sets = pd.read_csv(subjects_features_path, delimiter=";")

    # lists for holding the clustering results
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

                    # find the subject id in the subject features df to get the feature set for that subject
                    if subject_folder in feature_sets[SUBJECT_ID].values:

                        subject_feature_set_str = \
                            feature_sets.loc[feature_sets[SUBJECT_ID] == subject_folder, FEATURE_SET].values[0]

                        # Convert feature_set string into a list of strings
                        subject_feature_set = _parse_feature_set(subject_feature_set_str)

                        if subject_feature_set:
                            # Cluster the subject
                            ari, nmi = cluster_subject_all_activities(dataset_path, clustering_model, nr_clusters,
                                                                      subject_feature_set, subject_folder)

                            # # train test split
                            # train_set, test_set = load.train_test_split(dataset_path, 0.8, 0.2)


                            # # x, y split
                            # x_train = train_set.drop([CLASS, SUBCLASS], axis=1)
                            # y_train = train_set[CLASS]
                            # x_test = test_set.drop([CLASS, SUBCLASS], axis=1)
                            # y_test = test_set[CLASS]
                            #
                            # # try random forest
                            # accuracy_score = random_forest_classifier(x_train, x_test, y_train, y_test)

                            results.append({
                                "Subject ID": subject_folder,
                                "ARI": ari,
                                "NMI": nmi,
                                # "Random Forest acc": accuracy_score
                            })
                            # Inform user
                            print(f"Clustering results for subject: {subject_folder}")
                            # print(f"Feature set used: {subject_feature_set_str}")
                            print(
                                f"Adjusted Rand Index: {ari}; Normalized Mutual Information: {nmi}\n")
                        else:
                            print(f"Failed to parse feature set for subject {subject_folder}. Skipping.")
                    else:
                        raise ValueError(f"Subject ID {subject_folder} not found in the feature sets CSV.")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")

    # Create DataFrame from results and save to Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(results_path, "clustering_results_kmeans_rg_all_watch_phone.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

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

