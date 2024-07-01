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


def general_model_clustering(main_path: str, subfolder_name: str, clustering_model: str,
                             feature_set: List[str]) -> Tuple[float, float, float]:
    # load all subjects into a dataframe
    all_subjects_df = load.load_train_test_subjects(main_path, subfolder_name)

    # Check if all features in the feature set exist in the dataframe columns
    missing_features = [feature for feature in feature_set if feature not in all_subjects_df.columns]
    if missing_features:
        raise ValueError(f"The following features are not in the dataset columns: {missing_features}")

    # get only the wanted features
    all_subjects_df = all_subjects_df[feature_set]

    # normalize features
    all_subjects_df = _normalize_features(all_subjects_df)

    # Initialize evaluation metrics accumulators
    total_rand_index = 0.0
    total_adj_rand_index = 0.0
    total_norm_mutual_info = 0.0
    num_subjects = len(all_subjects_df['subject'].unique())

    # Perform leave-one-out cross-validation
    for i, test_subject in enumerate(all_subjects_df['subject'].unique(), 1):
        print(f"Testing with subject {test_subject} (Iteration {i}/{num_subjects})")

        # Split into train and test data
        train_subjects_df = all_subjects_df[all_subjects_df['subject'] != test_subject].drop(['class', 'subclass', 'subject'], axis=1)
        test_subjects_df = all_subjects_df[all_subjects_df['subject'] == test_subject].drop(['class', 'subclass', 'subject'], axis=1)
        true_labels = test_subjects_df['class']

        # Perform clustering based on the selected model
        if clustering_model == KMEANS:
            labels = models.kmeans_model(train_subjects_df, test_subjects_df, n_clusters=3)
        elif clustering_model == AGGLOMERATIVE:
            labels = models.agglomerative_clustering_model(train_subjects_df, test_subjects_df, n_clusters=3)
        elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
            labels = models.gaussian_mixture_model(train_subjects_df, test_subjects_df, n_components=3)
        elif clustering_model == DBSCAN:
            labels = models.dbscan_model(train_subjects_df, test_subjects_df, 0.4, 10)
        elif clustering_model == BIRCH:
            labels = models.birch_model(train_subjects_df, test_subjects_df, n_clusters=3)
        else:
            raise ValueError(f"The model {clustering_model} is not supported. "
                             f"Supported models are: {SUPPORTED_MODELS}")

        # Evaluate clustering
        rand_index, adj_rand_index, norm_mutual_info = metrics.evaluate_clustering(true_labels, labels)

        # Accumulate metrics
        total_rand_index += rand_index
        total_adj_rand_index += adj_rand_index
        total_norm_mutual_info += norm_mutual_info

    # Calculate average metrics
    avg_rand_index = total_rand_index / num_subjects
    avg_adj_rand_index = total_adj_rand_index / num_subjects
    avg_norm_mutual_info = total_norm_mutual_info / num_subjects

    print(f"Average clustering results over {num_subjects} subjects:\n"
          f"Avg Rand Index: {avg_rand_index}\n"
          f"Avg Adjusted Rand Index: {avg_adj_rand_index}\n"
          f"Avg Normalized Mutual Information: {avg_norm_mutual_info}")

    return avg_rand_index, avg_adj_rand_index, avg_norm_mutual_info


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
