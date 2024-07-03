# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

import load
import metrics
from .common import normalize_features, cluster_data


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def subject_specific_clustering(df: pd.DataFrame, clustering_model: str, feature_set: str, train_size: float = 0.7,
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
    labels = cluster_data(clustering_model, train_set, test_set, n_clusters=3)

    # evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, labels)

    return ri, ari, nmi
