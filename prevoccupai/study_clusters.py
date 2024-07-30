# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #


from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------------------------------------------------------------- #
# functions
# ------------------------------------------------------------------------------------------------------------------- #

def elbow_method(df):
    wcss_list = []
    n_clusters = range(1, 11)
    for k in n_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        wcss_list.append(kmeans.inertia_)
    knee = KneeLocator(n_clusters, wcss_list, curve='convex', direction='decreasing')
    elbow_point = knee.elbow

    return elbow_point, wcss_list


def silhouette_analysis(df):  # working
    silhouette_scores = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    max_s = max(silhouette_scores)
    best_k = k_values[silhouette_scores.index(max_s)]

    return max_s, best_k, silhouette_scores


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #