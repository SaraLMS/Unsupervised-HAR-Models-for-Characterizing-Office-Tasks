# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


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