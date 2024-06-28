# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import load
import models
import metrics
from typing import List, Tuple
from constants import KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH, SUPPORTED_MODELS


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def cluster_all_subjects(main_path: str, subfolder_name: str, clustering_model: str,
                         feature_set: List[str], train_size: float = 0.7) -> Tuple[float, float, float]:
    # split train and test subjects
    train_subjects_df, test_subjects_df = load.load_train_test_subjects(main_path, subfolder_name, train_size)

    # get true labels from the test subjects
    true_labels = test_subjects_df['class']

    # drop class, subclass and subject columns
    train_subjects_df = train_subjects_df.drop(['class', 'subclass', 'subject'], axis=1)
    test_subjects_df = test_subjects_df.drop(['class', 'subclass', 'subject'], axis=1)

    # get only the wanted features
    train_subjects_df = train_subjects_df[feature_set]
    test_subjects_df = test_subjects_df[feature_set]

    # normalize features
    train_subjects_df = _normalize_features(train_subjects_df)
    test_subjects_df = _normalize_features(test_subjects_df)

    if clustering_model == KMEANS:
        # kmeans feature_engineering
        labels = models.kmeans_model(train_subjects_df, test_subjects_df, n_clusters=3)

    elif clustering_model == AGGLOMERATIVE:
        # agglomerative feature_engineering
        labels = models.agglomerative_clustering_model(train_subjects_df, train_subjects_df, n_clusters=3)

    elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
        # gaussian mixture model
        labels = models.gaussian_mixture_model(train_subjects_df, train_subjects_df, n_components=3)

    elif clustering_model == DBSCAN:
        # DBSCAN feature_engineering
        labels = models.dbscan_model(train_subjects_df, train_subjects_df, 0.4, 10)

    elif clustering_model == BIRCH:
        # Birch feature_engineering
        labels = models.birch_model(train_subjects_df, test_subjects_df, n_clusters=3)

    else:
        raise ValueError(f"The model {clustering_model} is not supported. "
                         f"Supported models are: {SUPPORTED_MODELS}")

    # evaluate clustering
    rand_index, adj_rand_index, norm_mutual_info = metrics.evaluate_clustering(true_labels, labels)
    print(
        f"General model clustering results:\nRand Index: {rand_index}\nAdjusted Rand Index: {adj_rand_index}\nNormalized Mutual Information: {norm_mutual_info}")

    return rand_index, adj_rand_index, norm_mutual_info


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


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
