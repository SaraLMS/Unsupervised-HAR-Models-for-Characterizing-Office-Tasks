# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import models
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH, SUPPORTED_MODELS


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def normalize_features(x: pd.DataFrame) -> pd.DataFrame:
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


def cluster_data(clustering_model: str, train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int) -> pd.Series:
    # Perform clustering based on the selected model
    if clustering_model == KMEANS:
        labels = models.kmeans_model(train_set, test_set, n_clusters)
    elif clustering_model == AGGLOMERATIVE:
        labels = models.agglomerative_clustering_model(train_set, test_set, n_clusters)
    elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
        labels = models.gaussian_mixture_model(train_set, test_set, n_clusters)
    elif clustering_model == DBSCAN:
        labels = models.dbscan_model(train_set, test_set, 0.4, 10)
    elif clustering_model == BIRCH:
        labels = models.birch_model(train_set, test_set, n_clusters)
    else:
        raise ValueError(f"The model {clustering_model} is not supported. "
                         f"Supported models are: {SUPPORTED_MODELS}")
    return labels
