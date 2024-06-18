# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from collections import Counter
from itertools import combinations
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score

from feature_engineering.models import kmeans_model, agglomerative_clustering_model, gaussian_mixture_model, \
    dbscan_model, birch_model
from feature_engineering.split_train_test import train_test_split
from load.load_sync_data import load_data_from_csv
from parser.check_create_directories import create_dir

KMEANS = "kmeans"
AGGLOMERATIVE = "agglomerative"
GAUSSIAN_MIXTURE_MODEL = "gmm"
DBSCAN = "dbscan"
BIRCH = "birch"
SUPPORTED_MODELS = [KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH]


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def feature_selector(dataset_path: str, variance_threshold: float, n_iterations: int, clustering_model: str,
                     output_path: str,
                     folder_name: str = "features_kmeans_plots", save_plots: bool = False) -> Tuple[
    List[str], List[float]]:
    """
    Splits the dataset into train and test and returns the features sets that give the best clustering results
    for the train set as well as the rand index of the respective feature set.

    The first step of the feature selection is removing the low variance features. Drops the features with variance
    lower than variance_threshold. Then, shuffles the remaining features and iteratively adds features. The feature is
    removed if the accuracy (rand index) doesn't improve. This process generated a plot with the x-axis being the features
    added in each iteration and the y-axis the scores for the rand index, adjusted rand index and normalized mutual info.
    The plot is generated and saved if save_plots is set to True.

    This process is repeated n_iterations times and with every iteration, the features are shuffled so that different
    combinations of features can be tested.

    :param dataset_path: str
    Path to the file containing the features extrated (columns) and data instances (rows).

    :param variance_threshold: float
    Minimum variance value. Features with a training-set variance lower than this threshold will be removed.

    :param n_iterations: int
    Number of times the feature selection method is repeated.

    :param clustering_model: str
    Unsupervised learning model used to select the best features. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model
        "dbscan": DBSCAN. Needs parameter search - not implemented
        "birch": Birch clustering algorithm

    :param output_path: str
    Path to the main folder in which the plots should be saved

    :param folder_name: str
    Name of the folder in which to store the plots

    :param save_plots: bool (default = True)
    If True, Save the plots of each iteration of the feature selection process. Don't save if False.
    :return:
    """
    # load dataset from csv file
    df = load_data_from_csv(dataset_path)

    # generate output path to save the plots if it doesn't exist
    output_path = create_dir(output_path, folder_name)

    # train test split
    train_set, _ = train_test_split(df, 0.8, 0.2)

    # get the true (class) labels
    true_labels = train_set['class']

    # drop class and subclass column
    train_set = train_set.drop(['class', 'subclass'], axis=1)

    # remove class and subclass column and standardize features
    train_set = _standardize_features(train_set)

    # drop features with variance lower than variance_threshold
    train_set = _drop_low_variance_features(train_set, variance_threshold)
    print(f"Columns after dropping low variance features: {train_set.columns.tolist()}")
    train_set = _remove_collinear_features(train_set, 0.97)
    print(f"Columns after dropping low variance features: {train_set.columns.tolist()}")

    feature_sets = []
    feature_sets_accur = []

    for i in range(1, n_iterations + 1):
        # Reset the best accuracy for each iteration
        best_ri = 0

        # Shuffle the column names at the beginning of each iteration
        shuffled_features = _shuffle_column_names(train_set)
        print(f"Shuffled Features (Iteration {i}): {shuffled_features}")

        # Reset the feature list for each iteration
        iter_feature_list = []

        # Temporary lists for storing the best features and metrics of the current iteration
        best_features = []
        accur_ri = []
        adj_rand_scores = []
        norm_mutual_infos = []

        # Cycle over the shuffled columns of the dataframe
        for feature in shuffled_features:
            # Add the feature to the feature list
            iter_feature_list.append(feature)

            # Get the corresponding columns
            features_train = train_set[iter_feature_list]

            if clustering_model == KMEANS:
                # kmeans feature_engineering
                labels = kmeans_model(features_train, n_clusters=3)

            elif clustering_model == AGGLOMERATIVE:
                # agglomerative feature_engineering
                labels = agglomerative_clustering_model(features_train, n_clusters=3)

            elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
                # gaussian mixture model
                labels = gaussian_mixture_model(features_train, n_components=3)

            elif clustering_model == DBSCAN:
                # DBSCAN feature_engineering
                labels = dbscan_model(features_train, 0.4, 10)

            elif clustering_model == BIRCH:
                # Birch feature_engineering
                labels = birch_model(features_train, n_clusters=3)

            else:
                raise ValueError(f"The model {clustering_model} is not supported. "
                                 f"Supported models are: {SUPPORTED_MODELS}")

            # Evaluate clustering with this feature set
            ri, ari, nmi = _evaluate_clustering(true_labels, labels)

            # if the Rand Index does not improve remove feature
            if ri <= best_ri:
                iter_feature_list.remove(feature)

            # if Rand Index improves add the feature to the feature list
            else:
                best_ri = ri
                best_features.append(iter_feature_list.copy())

                # Add results to the respective lists
                accur_ri.append(ri)
                adj_rand_scores.append(ari)
                norm_mutual_infos.append(nmi)

        print(f"Iteration {i}: Best Features - {best_features}")

        # X-axis will show the feature sets
        feature_names = ['\n'.join(features) for features in best_features]
        best_features_list = best_features[-1]
        highest_accuracy = accur_ri[-1]

        print(best_features_list)

        if best_features_list not in feature_sets:

            feature_sets.append(best_features_list)
            feature_sets_accur.append(highest_accuracy)
            file_path = f"{output_path}/feature_set{i}_results.png"

            # save plots if save_plots == True
            if save_plots:
                # plot clustering results for that feature set
                _save_plot_clustering_results(feature_names, accur_ri, adj_rand_scores, norm_mutual_infos, file_path,
                                              clustering_model)
                # inform user
                print(f"Plot saved with the following features: {best_features_list}")
        else:
            # inform user
            print(f"Feature combination already tested.")

    return feature_sets, feature_sets_accur


def get_all_subjects_best_features(main_path: str, features_folder_name: str, variance_threshold: float,
                                   n_iterations: int, clustering_model: str):
    subjects_dict = {}

    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)
        print(f"Selecting best features for subject: {subject_folder}")

        for sub_folder in os.listdir(subject_folder_path):
            sub_folder_path = os.path.join(subject_folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):

                features_folder_path = os.path.join(sub_folder_path, features_folder_name)
                if os.path.isdir(features_folder_path):
                    feature_files = os.listdir(features_folder_path)

                    if len(feature_files) == 1:
                        dataset_path = os.path.join(features_folder_path, feature_files[0])

                        # Get the best feature sets for the subject
                        feature_sets, accuracies = feature_selector(dataset_path, variance_threshold, n_iterations,
                                                                    clustering_model, sub_folder_path)

                        # Filter for the best feature sets and their accuracies
                        best_feature_sets, best_acc = _filter_best_feature_sets(feature_sets, accuracies)

                        print("#########################################################################")
                        print(f"SUBJECT: {subject_folder}")
                        print("#########################################################################")
                        print(f"Feature sets with the highest accuracies:")
                        for feat, acc in zip(best_feature_sets, best_acc):
                            print(f"Features: {feat} \n Rand Index: {acc}")

                        most_common_n_features = _find_most_common_features_in_best_sets(best_feature_sets)
                        print(f"Feature frequency in best feature sets: {most_common_n_features} \n")

                        # store only the feature names without the counter
                        most_common_n_features = [feature for feature, count in most_common_n_features]
                        subjects_dict[subject_folder] = most_common_n_features

                    else:
                        raise ValueError(f"Too many files: {len(feature_files)} files. Only one dataset per folder.")
                else:
                    print(f"No features folder found in: {sub_folder_path}")

    return subjects_dict


def get_top_features_across_all_subjects(subjects_dict):
    # Aggregate the most common features across all subjects
    feature_list = _aggregate_most_common_features(subjects_dict)

    # Count the frequency of each feature
    feature_counter = Counter(feature_list)
    print("Best feature occurrence across all subjects:")
    for feature, count in feature_counter.items():
        print(f"{feature}: {count} subjects")

    # Return a list of the most common features
    most_common_features = [feature for feature, count in feature_counter.most_common()]
    print(f"Final feature set for all subjects: {most_common_features}")
    return most_common_features


# ------------------------------------------------------------------------------------------------------------------- #
# Private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _save_plot_clustering_results(feature_names: List[str], accur_ri: List[float], adj_rand_scores: List[float],
                                  norm_mutual_infos: List[float], file_path: str, clustering_model: str) -> None:
    # generate plot
    plt.figure(figsize=(14, 7))
    plt.plot(feature_names, accur_ri, marker='o', linestyle='-', color='#D36135', label='Rand Index')
    plt.plot(feature_names, adj_rand_scores, marker='x', linestyle='-', color='#386FA4',
             label='Adjusted Rand Index')
    plt.plot(feature_names, norm_mutual_infos, marker='s', linestyle='-', color='#4D9078',
             label='Normalized Mutual Info')
    plt.xlabel('Feature Sets')
    plt.ylabel('Scores')
    plt.title(clustering_model.replace('_', ' '))
    plt.xticks(rotation=0)  # Set rotation to 0 to keep the labels horizontal
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def _standardize_features(x: pd.DataFrame):
    # get new column names
    x_column_names = x.columns

    # standardize the features
    x = MinMaxScaler().fit_transform(x)

    # new dataframe without class and subclass column
    x = pd.DataFrame(x, columns=x_column_names)

    return x


def _evaluate_clustering(true_labels: pd.Series, predicted_labels: pd.Series):
    # calculate feature_engineering accuracy
    rand_index = np.round(rand_score(true_labels, predicted_labels), 4)

    # rand index adjusted for chance
    adjusted_rand_index = np.round(adjusted_rand_score(true_labels, predicted_labels), 4)

    # mutual information
    normalized_mutual_info = np.round(normalized_mutual_info_score(true_labels, predicted_labels), 4)

    return rand_index, adjusted_rand_index, normalized_mutual_info


def _drop_low_variance_features(x: pd.DataFrame, variance_threshold: float):
    # check low variance features
    var_thr = VarianceThreshold(threshold=variance_threshold)
    var_thr.fit(x)

    # low variance columns
    concol = [column for column in x.columns
              if column not in x.columns[var_thr.get_support()]]

    # drop columns
    x = x.drop(concol, axis=1)

    return x


def _remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x


def _shuffle_column_names(x: pd.DataFrame):
    column_names = list(x.columns)
    random.shuffle(column_names)
    return column_names


def _remove_first_letter(feature_sets: List[List[str]]) -> List[List[str]]:
    return [[feature[1:] for feature in feature_set] for feature_set in feature_sets]


def _filter_best_feature_sets(feature_sets, feature_sets_accur):
    # Find the highest accuracy
    highest_accuracy = max(feature_sets_accur)

    # Filter feature sets and accuracies that have the highest accuracy
    best_feature_sets = [feature_sets[i] for i, accur in enumerate(feature_sets_accur) if accur == highest_accuracy]
    best_accuracies = [accur for accur in feature_sets_accur if accur == highest_accuracy]

    return best_feature_sets, best_accuracies


def _find_most_common_features_in_best_sets(best_feature_sets, n=4):
    # Flatten the list of lists
    flat_feature_sets = [feature for feature_set in best_feature_sets for feature in feature_set]
    # Count the frequency of each feature
    feature_counter = Counter(flat_feature_sets)
    # Get the most common features and their counts
    most_common_features = feature_counter.most_common(n)
    return most_common_features


def _aggregate_most_common_features(subjects_dict):
    all_most_common_features = []
    for features in subjects_dict.values():
        all_most_common_features.extend(features)
    return all_most_common_features

