# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, DBSCAN
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------- #
# constants
# ------------------------------------------------------------------------------------------------------------------- #

# muda isto para o nome da tua class column
CLASS = "class"

# isto prob nao é necessario para ti
SUBJECT = "subject"
SUBCLASS = "subclass"

KMEANS = "kmeans"
AGGLOMERATIVE = "agglomerative"
GAUSSIAN_MIXTURE_MODEL = "gmm"
DBSCAN = "dbscan"
BIRCH = "birch"
SUPPORTED_MODELS = [KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def feature_selector(train_set: pd.DataFrame, variance_threshold: float, correlation_threshold: float,
                     n_iterations: int, clustering_model: str,
                     output_path: str, folder_name: str = "phone_features_kmeans_plots",
                     save_plots: bool = False) -> Tuple[List[List[str]], List[float], List[float], List[float]]:
    """
    This function returns the features sets that give the best clustering results for the train set
    as well as the rand index of the respective feature sets.
    # TODO (1) (2) (3)

    The first step of the feature selection is removing the low variance and highly correlated features.
    Then, shuffles the remaining features and iteratively adds one feature at a time. If by adding the feature, the
    rand index (accuracy) doesn't improve, that feature is removed from the feature set.
    This process generates a plot with the x-axis being the features added in each iteration and the y-axis the scores
    for the rand index, adjusted rand index and normalized mutual information.

    The plot is generated and saved if save_plots is set to True.

    This process is repeated n_iterations times and with every iteration, the column names (feature names) are shuffled
    so that different combinations of features can be tested.

    :param train_set: pd.DataFrame
    Train set containing the features extracted (columns) and data instances (rows).

    :param variance_threshold: float
    Minimum variance value. Features with variance lower than this threshold will be removed.

    :param correlation_threshold: float
    Maximum correlation value. Removes features with a correlation higher or equal to this threshold

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

    :param folder_name: (optional) str
    Name of the folder in which to store the plots. Default: "phone_features_kmeans_plots".

    :param save_plots: bool (default = True)
    If True, saves the plots of each iteration of the feature selection process. Don't save if False.

    :return: Tuple[List[List[str]], List[float]]
    List of feature sets and another list with the rand index of each feature set
    """
    # TODO CHECK MODELS
    # get the true (class) labels
    true_labels = train_set[CLASS]

    # # drop class and subclass column
    # train_set = train_set.drop(['class', 'subclass'], axis=1)
    train_set = train_set.drop([CLASS], axis=1)

    # remove subject column if exists
    if SUBJECT in train_set.columns:
        train_set = train_set.drop([SUBJECT], axis=1)

    if SUBCLASS in train_set.columns:
        train_set = train_set.drop([SUBCLASS], axis=1)

    # (1) scale features
    # TODO EXPLAIN IN DOCSTRING NORM
    train_set = _normalize_features(train_set)

    # drop features with variance lower than variance_threshold
    train_set = _drop_low_variance_features(train_set, variance_threshold)

    # drop correlated features
    train_set = _remove_collinear_features(train_set, correlation_threshold)

    if save_plots:
        # generate output path to save the plots if it doesn't exist
        output_path = create_dir(output_path, folder_name)

    # lists to save the features sets and the respective scores
    feature_sets = []
    feature_sets_ri = []
    feature_sets_ari = []
    feature_sets_nmi = []

    for iteration in range(1, n_iterations + 1):
        # Reset the best accuracy for each iteration
        best_ri = 0

        # Shuffle the column names at the beginning of each iteration
        shuffled_features = _shuffle_column_names(train_set)
        # print(f"Shuffled Features (Iteration {iteration}): {shuffled_features}")

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

            # cluster the data
            pred_labels = cluster_data(clustering_model, features_train, features_train, n_clusters=3)

            # Evaluate clustering with this feature set
            ri, ari, nmi = evaluate_clustering(true_labels, pred_labels)

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

        print(f"Iteration {iteration}: Best Features - {best_features}\n")

        # X-axis of the plot will show the features
        feature_names = ['\n'.join(features) for features in best_features]

        # last position of the list has the final feature set and correspondent rand index
        # inform user
        print(f"Best features list: {best_features[-1]}\n"
              f"Rand Index: {accur_ri[-1]}\n"
              f"Adjusted Rand Index: {adj_rand_scores[-1]}\n"
              f"Normalized Mutual Information: {norm_mutual_infos[-1]}")

        if best_features[-1] not in feature_sets:

            # save the feature set
            feature_sets.append(best_features[-1])

            # save the scores of this feature set
            feature_sets_ri.append(accur_ri[-1])
            feature_sets_ari.append(adj_rand_scores[-1])
            feature_sets_nmi.append(norm_mutual_infos[-1])

            if save_plots:
                # generate filepath to store the plot
                file_path = f"{output_path}/feature_set{iteration}_results.png"

                # plot clustering results for that feature set
                _save_plot_clustering_results(feature_names, accur_ri, adj_rand_scores, norm_mutual_infos, file_path,
                                              clustering_model)
                # inform user
                print(f"Plot saved with the following features: {best_features[-1]}")
        else:
            # inform user
            print(f"Feature combination already tested.")

    return feature_sets, feature_sets_ri, feature_sets_ari, feature_sets_nmi


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


def _remove_collinear_features(x: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    https://www.kaggle.com/code/oldwine357/removing-highly-correlated-features

    Remove collinear features in a dataframe with a correlation coefficient
    greater than the threshold. Removing collinear features can help a model
    to generalize and improves the interpretability of the model.

    :param x: pd.DataFrame
    A pandas DataFrame containing the features

    :param threshold: float
    features with correlations greater than this value are removed

    :return: pd.DataFrame
    A pandas DataFrame that contains only the non-highly-collinear features
    """

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


def _shuffle_column_names(x: pd.DataFrame) -> List[str]:
    """
    Shuffle the column names of a dataframe.

    :param x: pd.DataFrame
    Pandas dataframe containing the data

    :return: List[str]
    List containing the shuffled column names
    """
    column_names = list(x.columns)
    random.shuffle(column_names)
    return column_names

def create_dir(path: str, folder_name: str) -> str:
    """
    creates a new directory in the specified path
    :param path: the path in which the folder_name should be created
    :param folder_name: the name of the folder that should be created
    :return: the full path to the created folder
    """

    # join path and folder
    new_path = os.path.join(path, folder_name)

    # check if the folder does not exist yet
    if not os.path.exists(new_path):
        # create the folder
        os.makedirs(new_path)

    return new_path

def cluster_data(clustering_model: str, train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int) -> pd.Series:
    # Perform clustering based on the selected model
    if clustering_model == KMEANS:
        labels = kmeans_model(train_set, test_set, n_clusters)
    elif clustering_model == AGGLOMERATIVE:
        labels = agglomerative_clustering_model(train_set, test_set, n_clusters)
    elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
        labels = gaussian_mixture_model(train_set, test_set, n_clusters)
    elif clustering_model == DBSCAN:
        labels = dbscan_model(train_set, test_set, 0.4, 10)
    elif clustering_model == BIRCH:
        labels = birch_model(train_set, test_set, n_clusters)
    else:
        raise ValueError(f"The model {clustering_model} is not supported. "
                         f"Supported models are: {SUPPORTED_MODELS}")
    return labels


def kmeans_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int) -> pd.Series:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(train_set)
    labels_kmeans = kmeans.predict(test_set)
    # predicted_labels = KMeans(n_clusters, random_state).fit_predict(df)
    return labels_kmeans


def gaussian_mixture_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_components: int) -> pd.Series:
    gaussian_mixture = GaussianMixture(n_components=n_components, reg_covar=1e-5, random_state=42, max_iter=500).fit(train_set)
    labels_gmm = gaussian_mixture.predict(test_set)
    return labels_gmm


def agglomerative_clustering_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int) -> pd.Series:
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(train_set)
    labels_agg = agg_clustering.predict(test_set)
    return labels_agg


def dbscan_model(train_set: pd.DataFrame, test_set: pd.DataFrame, epsilon: float, min_points: int) -> pd.Series:
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points).fit(train_set)
    labels_dbscan = dbscan.predict(test_set)
    return labels_dbscan


def birch_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int) -> pd.Series:
    birch = Birch(n_clusters=n_clusters, threshold=0.1).fit(train_set)
    labels_birch = birch.predict(test_set)
    return labels_birch


def evaluate_clustering(true_labels: pd.Series, predicted_labels: pd.Series) -> Tuple[float, float, float]:
    """
    Evaluates clustering performance using multiple metrics.

    This function computes the Rand Index, Adjusted Rand Index, and Normalized Mutual Information
    to evaluate the performance of a clustering algorithm based on the true and predicted labels.

    :param true_labels: pd.Series
    The ground truth labels for the data points.

    :param predicted_labels: pd.Series
    The labels predicted by the clustering algorithm.

    :return: Tuple[float, float, float]
    Rand Index, Adjusted Rand Index, and Normalized Mutual Information scores.
    """
    # calculate feature_engineering accuracy
    rand_index = np.round(rand_score(true_labels, predicted_labels), 2)

    # rand index adjusted for chance
    adjusted_rand_index = np.round(adjusted_rand_score(true_labels, predicted_labels), 2)

    # mutual information
    normalized_mutual_info = np.round(normalized_mutual_info_score(true_labels, predicted_labels), 2)

    return rand_index, adjusted_rand_index, normalized_mutual_info