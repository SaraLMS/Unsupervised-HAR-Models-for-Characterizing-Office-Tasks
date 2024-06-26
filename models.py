# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def kmeans_model(train_set: pd.DataFrame, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(train_set)

    return labels_kmeans


def gaussian_mixture_model(df: pd.DataFrame, n_components: int):
    gaussian_mixture = GaussianMixture(n_components=n_components, reg_covar=1e-5, random_state=42, max_iter=500)
    labels_gmm = gaussian_mixture.fit_predict(df)
    return labels_gmm


def agglomerative_clustering_model(df: pd.DataFrame, n_clusters):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg_clustering.fit_predict(df)
    return labels_agg


def dbscan_model(df: pd.DataFrame, epsilon: float, min_points: int):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    labels_dbscan = dbscan.fit_predict(df)
    return labels_dbscan


def birch_model(df: pd.DataFrame, n_clusters: int):
    birch = Birch(n_clusters=n_clusters, threshold=0.1)
    labels_birch = birch.fit_predict(df)
    return labels_birch
