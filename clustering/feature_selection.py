# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from clustering.metrics import evaluate_clustering
from clustering.models import kmeans_model, agglomerative_clustering_model, gaussian_mixture_model, dbscan_model, \
    birch_model


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def feature_selector(df: pd.DataFrame):
    # get the true (class) labels
    true_labels = df['class']

    # remove class and subclass column and standardize features
    x = _standardize_features(df)

    # drop low variance features
    x = _drop_low_variance_features(x, variance_threshold=0.1)

    # list for holding the features in each iteration of the for loop
    feature_list = []

    # list for holding the lists of features
    best_features = []

    # variable fo holding the best accuracy for each iteration
    best_ri_kmeans=0

    # lists for holding the accuracy of each model
    accur_kmeans = []
    accur_agg = []
    accur_gmm = []
    accur_dbscan = []
    accur_birch = []

    # cycle over the columns of the dataframe
    for feature in x.columns:
        # add the feature to the feature list
        feature_list.append(feature)

        # get the features for clustering
        features = x[feature_list]

        # # add feature set to teh feature set list
        # feature_set_list.append(len(feature_list))

        # kmeans clustering
        labels_kmeans = kmeans_model(features, n_clusters=3)

        # agglomerative clustering
        labels_agg = agglomerative_clustering_model(features, n_clusters=3)

        # gaussian mixture model
        labels_gmm = gaussian_mixture_model(features, n_components=3)

        # DBSCAN - NEEDS PARAMETER SEARCH!!!!!!!!!!!!!!!!!!!!!!!!!
        labels_dbscan = dbscan_model(features, epsilon=0.4, min_points=10)

        # Birch clustering
        labels_birch = birch_model(features, n_clusters=3)

        # evaluate clustering
        ri_kmeans, ari_kmeans, nmi_kmeans = evaluate_clustering(features, true_labels, labels_kmeans)
        ri_agg, ari_agg, nmi_agg = evaluate_clustering(features, true_labels, labels_agg)
        ri_gmm, ari_gmm, nmi_gmm = evaluate_clustering(features, true_labels, labels_gmm)
        ri_dbscan, ari_dbscan, nmi_dbscan = evaluate_clustering(features, true_labels, labels_dbscan)
        ri_birch, ari_birch, nmi_birch = evaluate_clustering(features, true_labels, labels_birch)

        if ri_kmeans < best_ri_kmeans:
            feature_list.remove(feature)
        else:
            best_ri_kmeans=ri_kmeans

            best_features.append(feature_list.copy())
            # add accuracy to the accuracy list
            accur_kmeans.append(ri_kmeans)

        accur_gmm.append(ri_gmm)
        accur_agg.append(ri_agg)
        accur_dbscan.append(ri_dbscan)
        accur_birch.append(ri_birch)

    return best_features, accur_kmeans


def plot_results(best_features: List[str], accuracy_list: List[float]):
    print(best_features)
    num_features = [len(features) for features in best_features]
    plt.figure(figsize=(10, 6))
    plt.plot(num_features, accuracy_list, marker='o', linestyle='-', color='b', label='KMeans')
    plt.xlabel('Number of Features')
    plt.ylabel('Rand Index')
    plt.title('Clustering Accuracy vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------------------------------------------------------- #
# Private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _standardize_features(df: pd.DataFrame):
    # drop class and subclass column
    x = df.drop(['class', 'subclass'], axis=1)

    # get new column names
    x_column_names = x.columns

    # standardize the features
    x = MinMaxScaler().fit_transform(x)

    # new dataframe without class and subclass column
    x = pd.DataFrame(x, columns=x_column_names)

    return x


def _correlated_features(corr_matrix, threshold: float = 0.99):
    # Absolute value
    corr_matrix = corr_matrix.abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index and column name of features with correlation greater than threshold
    corr_features = [column for column in upper.columns if any(upper[column] > threshold)]

    return corr_features


def _drop_correlated_features(x: pd.DataFrame):
    # compute pairwise correlation of columns/features
    corr_pearson = x.corr()

    # get correlated features
    corr_features = _correlated_features(corr_pearson)

    # drop correlated features
    x = x.drop(corr_features, axis=1)

    return x


def _drop_low_variance_features(x: pd.DataFrame, variance_threshold: float):
    # check low variance features
    # keep the features that have atleast 90% variance
    var_thr = VarianceThreshold(threshold=variance_threshold)
    var_thr.fit(x)

    # low variance columns
    concol = [column for column in x.columns
              if column not in x.columns[var_thr.get_support()]]

    # drop columns
    x = x.drop(concol, axis=1)

    return x
