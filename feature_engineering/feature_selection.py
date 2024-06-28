# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from collections import Counter
from typing import List, Tuple, Dict, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import models
import load
import parser
import metrics
from constants import KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH, SUPPORTED_MODELS


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def feature_selector(train_set: pd.DataFrame, variance_threshold: float, n_iterations: int, clustering_model: str,
                     output_path: str, folder_name: str = "phone_features_kmeans_plots",
                     save_plots: bool = False) -> Tuple[List[List[str]], List[float]]:
    """
    Splits the dataset into train and test and returns the features sets that give the best clustering results
    for the train set as well as the rand index of the respective feature set.

    The first step of the feature selection is removing the low variance and highly correlated features.
    Then, shuffles the remaining features and iteratively adds features. The feature is removed if the accuracy
    (rand index) doesn't improve. This process generated a plot with the x-axis being the features added in each
    iteration and the y-axis the scores for the rand index, adjusted rand index and normalized mutual information.
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

    # get the true (class) labels
    true_labels = train_set['class']

    # drop class and subclass column
    train_set = train_set.drop(['class', 'subclass'], axis=1)

    # remove subject column if exists
    if "subject" in train_set.columns:
        train_set = train_set.drop(['subject'], axis=1)

    # scale features
    train_set = _normalize_features(train_set)

    # drop features with variance lower than variance_threshold
    train_set = _drop_low_variance_features(train_set, variance_threshold)

    # drop correlated features
    train_set = _remove_collinear_features(train_set, 0.99)

    if save_plots:
        # generate output path to save the plots if it doesn't exist
        output_path = parser.create_dir(output_path, folder_name)

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
                labels = models.kmeans_model(features_train, features_train, n_clusters=3)

            elif clustering_model == AGGLOMERATIVE:
                # agglomerative feature_engineering
                labels = models.agglomerative_clustering_model(features_train, features_train, n_clusters=3)

            elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
                # gaussian mixture model
                labels = models.gaussian_mixture_model(features_train, features_train, n_components=3)

            elif clustering_model == DBSCAN:
                # DBSCAN feature_engineering
                labels = models.dbscan_model(features_train, features_train, 0.4, 10)

            elif clustering_model == BIRCH:
                # Birch feature_engineering
                labels = models.birch_model(features_train, features_train, n_clusters=3)

            else:
                raise ValueError(f"The model {clustering_model} is not supported. "
                                 f"Supported models are: {SUPPORTED_MODELS}")

            # Evaluate clustering with this feature set
            ri, ari, nmi = metrics.evaluate_clustering(true_labels, labels)

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

        # print(f"Iteration {i}: Best Features - {best_features}\n")

        # X-axis of the plot will show the features
        feature_names = ['\n'.join(features) for features in best_features]

        # last position of the list has the final feature set and its rand index
        best_features_list = best_features[-1]
        highest_accuracy = accur_ri[-1]

        # inform user
        print(f"Best features list: {best_features_list}\n"
              f"Rand Index: {accur_ri[-1]}\n"
              f"Adjusted Rand Index: {adj_rand_scores[-1]}\n"
              f"Normalized Mutual Information: {norm_mutual_infos[-1]}")

        if best_features_list not in feature_sets:

            # save the feature set
            feature_sets.append(best_features_list)

            # save the rand index of this feature set
            feature_sets_accur.append(highest_accuracy)

            if save_plots:
                # generate filepath to store the plot
                file_path = f"{output_path}/feature_set{i}_results.png"

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
                                   n_iterations: int, clustering_model: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Apply the feature selection method in feature_selector for every subject. Filters the feature sets to get only
    the ones with the highest accuracy, then counts the n most common features. Returns a dictionary with the subjects
    and the lists with the n most common features in the best feature sets.

    The subjects diretories must be organized the following way:
    main_path/subjects_folders/features_folder (features_folder_name)/csv_file(one only)

    The output path to the plots is main_path/subjects_folders/subfolder, and the output folder name is set in feature
    selector as well as save_plots parameter, to save or not the plots.

    :param main_path: str
    Path to the main folder containing the subfolders of each subject

    :param features_folder_name: str
    Name of the folder containing the csv file with the extracted features

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

    :return: dict[str, List[str]]
    Dictionary where keys are the subject folder names and values are the Lists of the n most common features in the
    best feature sets.
    """
    subjects_dict = {}
    # TODO OUTPUT PATH was INCORRECT NOT TESTED THO
    # iterate through the folders of each subject
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)
        print(f"Selecting best features for subject: {subject_folder}")

        # iterate through the sub folders
        for sub_folder in os.listdir(subject_folder_path):

            if sub_folder == features_folder_name:

                features_folder_path = os.path.join(subject_folder_path, sub_folder)

                # list with the path to the csv file
                feature_files = os.listdir(features_folder_path)

                if len(feature_files) == 1:
                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, feature_files[0])

                    # train test split
                    train_set, _ = load.train_test_split(dataset_path, 0.8, 0.2)

                    # Get the best feature sets for the subject
                    feature_sets, accuracies = feature_selector(train_set, variance_threshold, n_iterations,
                                                                clustering_model, features_folder_name)

                    # Filter for the best feature sets and their accuracies
                    best_feature_sets, best_acc = _filter_best_feature_sets(feature_sets, accuracies)

                    print("#########################################################################")
                    print(f"SUBJECT: {subject_folder}")
                    print("#########################################################################")
                    print(f"Feature sets with the highest accuracies:")
                    for feat, acc in zip(best_feature_sets, best_acc):
                        print(f"Features: {feat} \n Rand Index: {acc}")

                    # Store the unique features in a set to avoid repetitions
                    unique_features = set()
                    unique_features_no_axis = set()
                    for feat_set in best_feature_sets:
                        unique_features.update(feat_set)
                        unique_features_no_axis.update([feature[1:] for feature in feat_set])

                    subjects_dict[subject_folder] = {
                        'unique_features': unique_features,
                        'unique_features_no_axis': unique_features_no_axis
                    }

                else:
                    raise ValueError(f"Too many files: {len(feature_files)} files. Only one dataset per folder.")

            else:
                ValueError(f"Folder name {features_folder_name} not found.")

    return subjects_dict


def get_top_features_across_all_subjects(subjects_dict: Dict[str, Dict[str, Set[str]]], top_n: int) -> Dict[
    str, List[str]]:
    """
    Aggregates and identifies the most common best features across all subjects.

    This function takes a dictionary where the keys are subject folder names and the values are dictionaries containing
    sets of unique features from the best feature sets of each subject (with and without axis). It then aggregates these
    features, counts their occurrences across all subjects, prints the feature occurrence counts, and returns a dictionary
    with two lists: the top n most common features with axis and the top n most common features without axis.

    :param subjects_dict: Dict[str, Dict[str, Set[str]]]
    Dictionary where keys are the subject folder names and values are dictionaries containing sets of unique features
    from the best feature sets.

    :param top_n: int
    Number of top features to return.

    :return: Dict[str, List[str]]
    Dictionary with keys 'features_with_axis' and 'features_without_axis', each mapping to a list of the top n most
    common features across all subjects.
    """
    features_with_axis = []
    features_without_axis = []

    for features in subjects_dict.values():
        features_with_axis.extend(features['unique_features'])
        features_without_axis.extend(features['unique_features_no_axis'])

    # Count the frequency of each feature
    feature_counter_with_axis = Counter(features_with_axis)
    feature_counter_without_axis = Counter(features_without_axis)
    #
    # print("Best feature occurrence across all subjects (with axis):")
    # for feature, count in feature_counter_with_axis.items():
    #     print(f"{feature}: {count} occurrences")
    #
    # print("\nBest feature occurrence across all subjects (without axis):")
    # for feature, count in feature_counter_without_axis.items():
    #     print(f"{feature}: {count} occurrences")

    # Select the top n most common features
    # most_common_features_with_axis = [feature for feature, count in feature_counter_with_axis.most_common(top_n)]
    # most_common_features_without_axis = [feature for feature, count in feature_counter_without_axis.most_common(top_n)]
    # Select the top n most common features with tie-breaking
    most_common_features_with_axis = _select_top_n_features_with_tie_breaking(feature_counter_with_axis, top_n)
    most_common_features_without_axis = _select_top_n_features_with_tie_breaking(feature_counter_without_axis, top_n)
    print(f"\nTop {top_n} features across all subjects (with axis): {most_common_features_with_axis}")
    print(f"\nTop {top_n} features across all subjects (without axis): {most_common_features_without_axis}")

    return {
        'features_with_axis': most_common_features_with_axis,
        'features_without_axis': most_common_features_without_axis
    }


def test_feature_set(feature_set: List[str], file_path: str, clustering_model: str) -> Tuple[float, float, float]:
    """
    Tests a specific set of features for clustering.

    This function evaluates the performance of a specified clustering model using a given set of features from a dataset.
    It calculates the Rand Index, Adjusted Rand Index, and Normalized Mutual Information scores.

    :param feature_set: List[str]
    List of features to be used for clustering.

    :param file_path: str
    Path to the dataset csv file

    :param clustering_model: str
    Unsupervised learning model used to test the feature set. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model
        "dbscan": DBSCAN. Needs parameter search - not implemented
        "birch": Birch clustering algorithm

    :return: Tuple[float, float, float]
    Rand index, adjusted rand index and normalized mutual information scores for that feature set.
    """

    # train test split
    train_set, _ = load.train_test_split(file_path, 0.8, 0.2)

    # Check if all features in the feature set exist in the dataframe columns
    missing_features = [feature for feature in feature_set if feature not in train_set.columns]
    if missing_features:
        raise ValueError(f"The following features are not in the dataset columns: {missing_features}")

    # get the true (class) labels
    true_labels = train_set['class']

    # drop class and subclass column
    train_set = train_set.drop(['class', 'subclass'], axis=1)

    # scale features
    train_set = _normalize_features(train_set)

    # get only the wanted features
    train_set = train_set[feature_set]

    if clustering_model == KMEANS:
        # kmeans feature_engineering
        labels = models.kmeans_model(train_set, train_set, n_clusters=3)

    elif clustering_model == AGGLOMERATIVE:
        # agglomerative feature_engineering
        labels = models.agglomerative_clustering_model(train_set, train_set, n_clusters=3)

    elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
        # gaussian mixture model
        labels = models.gaussian_mixture_model(train_set, train_set, n_components=3)

    elif clustering_model == DBSCAN:
        # DBSCAN feature_engineering
        labels = models.dbscan_model(train_set, train_set, 0.4, 10)

    elif clustering_model == BIRCH:
        # Birch feature_engineering
        labels = models.birch_model(train_set, train_set, n_clusters=3)

    else:
        raise ValueError(f"The model {clustering_model} is not supported. "
                         f"Supported models are: {SUPPORTED_MODELS}")

    # Evaluate clustering with this feature set
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, labels)

    return ri, ari, nmi


def test_feature_set_each_subject(main_path: str, features_folder_name: str, clustering_model: str,
                                  feature_set: List[str]) -> Tuple[float, float, float]:
    ri_list = []
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

                    ri, ari, nmi = test_feature_set(feature_set, dataset_path, clustering_model)
                    # print(f"RI: {ri}; ARI: {ari}; NMI: {nmi}")

                    # Append the results to the lists
                    ri_list.append(ri)
                    ari_list.append(ari)
                    nmi_list.append(nmi)

                else:
                    ValueError(f"Too many files: {len(feature_files)}")

            else:
                ValueError(f"Folder name {features_folder_name} not found.")

    return np.round(np.mean(ri_list), 2), np.round(np.mean(ari_list), 2), np.round(np.mean(nmi_list), 2)


def test_different_axis(subjects_dict: Dict[str, Dict[str, Set[str]]], main_path: str, features_folder_name: str,
                        clustering_model: str, top_n: int) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Tests the most common features without axis by adding different axes (x, y, z) and evaluates the performance.

    :param subjects_dict: Dict[str, Dict[str, Set[str]]]
    Dictionary where keys are the subject folder names and values are dictionaries with keys 'unique_features' and
    'unique_features_no_axis', and the values are sets of unique features and unique features without the axis prefix, respectively.

    :param main_path: str
    Path to the main folder containing the subfolders of each subject.

    :param features_folder_name: str
    Name of the folder containing the csv file with the extracted features.

    :param clustering_model: str
    Unsupervised learning model used to test the feature set.

    :param top_n: int
    Number of top features to select.

    :return: Dict[str, Dict[str, Tuple[float, float, float]]]
    Dictionary where keys are 'x', 'y', 'z' and values are dictionaries with keys as subject names and values as
    tuples of clustering scores.
    """
    results = {'x': {}, 'y': {}, 'z': {}}

    # Get the top n most common features without axis
    top_features_dict = get_top_features_across_all_subjects(subjects_dict, top_n)
    top_features_without_axis = top_features_dict['features_without_axis']

    for axis in ['x', 'y', 'z']:
        # Add the axis prefix to the top features without axis
        axis_feature_set = [f"{axis}{feature}" for feature in top_features_without_axis]

        print(f"\nTest: {axis_feature_set}\n")
        # Test the feature set with the axis prefix for each subject
        mean_ri, mean_ari, mean_nmi = test_feature_set_each_subject(main_path, features_folder_name, clustering_model,
                                                                    axis_feature_set)

        results[axis] = {'features': axis_feature_set, 'mean_ri': mean_ri, 'mean_ari': mean_ari, 'mean_nmi': mean_nmi}

    return results

# ------------------------------------------------------------------------------------------------------------------- #
# Private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _save_plot_clustering_results(feature_names: List[str], accur_ri: List[float], adj_rand_scores: List[float],
                                  norm_mutual_infos: List[float], file_path: str, clustering_model: str) -> None:
    """
    Generates and saves a plot of clustering results.

    This function creates a plot that visualizes the Rand Index, Adjusted Rand Index, and Normalized Mutual Information
    scores when adding feature by feature until the highest accuracy is reached.
    The plot is saved to the specified file path.

    :param feature_names: List[str]
    A list of feature set names corresponding to the x-axis labels of the plot.

    :param accur_ri: List[float]
    A list of Rand Index scores corresponding to the feature sets.

    :param adj_rand_scores: List[float]
    A list of Adjusted Rand Index scores corresponding to the feature sets.

    :param norm_mutual_infos: List[float]
    A list of Normalized Mutual Information scores corresponding to the feature sets.

    :param file_path: str
    The file path where the plot image will be saved.

    :param clustering_model: str
    Unsupervised learning model used to select the best features. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model
        "dbscan": DBSCAN. Needs parameter search - not implemented
        "birch": Birch clustering algorithm

    :return:
    """
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


def _normalize_features(x: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the features of a DataFrame using MinMaxscaler from sklearn.

    This function applies Min-Max scaling to the features of the input DataFrame, transforming the values to a range
    between 0 and 1.

    :param x: pd.DataFrame
    The input DataFrame containing the features to be standardized.

    :return: pd.DataFrame
    A DataFrame with the standardized feature values
    """
    # get new column names
    x_column_names = x.columns

    # standardize the features
    x = MinMaxScaler().fit_transform(x)

    # new dataframe without class and subclass column
    x = pd.DataFrame(x, columns=x_column_names)

    return x


def _drop_low_variance_features(x: pd.DataFrame, variance_threshold: float) -> pd.DataFrame:
    """
    Removes features with low variance from a DataFrame.

    This function drops features (columns) from the input DataFrame that have a variance below the specified threshold.
    It uses the VarianceThreshold from sklearn to identify and remove these low-variance features.

    :param x: pd.DataFrame
    The input DataFrame containing the features to be filtered.

    :param variance_threshold: float
    The threshold below which features will be considered low variance and removed.

    :return: pd.DataFrame
    A dataframe with low-variance features removed
    """
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
    # print('Removed Columns {}'.format(drops))
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


def _aggregate_features(subjects_dict: Dict[str, Set[str]]) -> List[str]:
    """
    Aggregates features from the best feature sets of all subjects into a single list.

    :param subjects_dict: Dict[str, Set[str]]
    Dictionary where keys are the subject folder names and values are sets of unique features from the best feature sets.

    :return: List[str]
    List of all features from the best feature sets of all subjects.
    """
    all_features = []
    for feature_set in subjects_dict.values():
        all_features.extend(feature_set)
    return all_features


def _select_top_n_features_with_tie_breaking(feature_counter: Counter, top_n: int) -> List[str]:
    """
    Selects the top n features from a Counter, with a tie-breaking strategy to randomly select among features with the same count.

    :param feature_counter: Counter
    Counter object with feature counts.

    :param top_n: int
    Number of top features to select.

    :return: List[str]
    List of the top n features.
    """
    # Sort features by count in descending order
    features_by_count = sorted(feature_counter.items(), key=lambda item: item[1], reverse=True)

    # List to store the top n features
    top_features = []

    # Track the current count group
    current_count = None

    # List to store features with the same count
    current_tie_group = []

    # Iterate through the sorted features by count
    for feature, count in features_by_count:

        # If top n features are already selected, stop the loop
        if len(top_features) >= top_n:
            break

        # Initialize the current count if it's None (first iteration)
        if current_count is None:
            current_count = count

        # If the count is the same as the current count, add to the tie group
        if count == current_count:
            current_tie_group.append(feature)
        else:
            if len(top_features) + len(current_tie_group) > top_n:
                top_features.extend(random.sample(current_tie_group, top_n - len(top_features)))
            else:
                top_features.extend(current_tie_group)
            current_tie_group = [feature]
            current_count = count

    if len(top_features) < top_n and current_tie_group:
        top_features.extend(random.sample(current_tie_group, top_n - len(top_features)))

    return top_features[:top_n]
# def _select_top_n_features_with_tie_breaking(feature_counter: Counter, top_n: int) -> List[str]:
#     """
#     Selects the top n features from a Counter, with a tie-breaking strategy to randomly select one feature among features with the same count.
#
#     :param feature_counter: Counter
#     Counter object with feature counts.
#
#     :param top_n: int
#     Number of top features to select.
#
#     :return: List[str]
#     List of the top n features.
#     """
#     # Sort features by count in descending order
#     features_by_count = sorted(feature_counter.items(), key=lambda item: item[1], reverse=True)
#
#     top_features = []  # List to store the top n features
#     current_count = None  # Track the current count group
#     current_tie_group = []  # List to store features with the same count
#
#     # Iterate through the sorted features by count
#     for feature, count in features_by_count:
#         # If we've already selected top n features, stop the loop
#         if len(top_features) >= top_n:
#             break
#
#         # Initialize the current count if it's None (first iteration)
#         if current_count is None:
#             current_count = count
#
#         # If the count is the same as the current count, add to the tie group
#         if count == current_count:
#             current_tie_group.append(feature)
#         else:
#             # If the count is different, resolve the tie group
#             if current_tie_group:
#                 # Randomly select one feature from the tie group and add it to the top features
#                 top_features.append(random.choice(current_tie_group))
#
#             # Reset the tie group for the new count
#             current_tie_group = [feature]
#             current_count = count
#
#     # After the loop, resolve any remaining tie group
#     if len(top_features) < top_n and current_tie_group:
#         top_features.append(random.choice(current_tie_group))
#
#     return top_features[:top_n]
