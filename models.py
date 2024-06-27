# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def kmeans_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(train_set)
    labels_kmeans = kmeans.predict(test_set)

    return labels_kmeans


def gaussian_mixture_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_components: int):
    gaussian_mixture = GaussianMixture(n_components=n_components, reg_covar=1e-5, random_state=42, max_iter=500).fit(train_set)
    labels_gmm = gaussian_mixture.predict(test_set)
    return labels_gmm


def agglomerative_clustering_model(train_set: pd.DataFrame, test_set:pd.DataFrame, n_clusters):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(train_set)
    labels_agg = agg_clustering.predict(test_set)
    return labels_agg


def dbscan_model(train_set: pd.DataFrame, test_set: pd.DataFrame, epsilon: float, min_points: int):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points).fit(train_set)
    labels_dbscan = dbscan.predict(test_set)
    return labels_dbscan


def birch_model(train_set: pd.DataFrame, test_set: pd.DataFrame, n_clusters: int):
    birch = Birch(n_clusters=n_clusters, threshold=0.1).fit(train_set)
    labels_birch = birch.predict(test_set)
    return labels_birch
