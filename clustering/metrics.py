# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def evaluate_clustering(x: pd.DataFrame, true_labels: pd.Series, predicted_labels: pd.Series):

    # calculate clustering accuracy
    rand_index = np.round(rand_score(true_labels, predicted_labels), 2)

    # rand index adjusted for chance
    adjusted_rand_index = np.round(adjusted_rand_score(true_labels, predicted_labels),2)

    # mutual information
    normalized_mutual_info = np.round(normalized_mutual_info_score(true_labels, predicted_labels), 2)

    # # silhouette coefficient
    # mean_silhouette = np.round(silhouette_score(x, predicted_labels),2)

    return rand_index, adjusted_rand_index, normalized_mutual_info


