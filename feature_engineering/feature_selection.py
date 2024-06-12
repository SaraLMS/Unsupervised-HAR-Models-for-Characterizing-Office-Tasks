# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from feature_engineering.models import kmeans_model, agglomerative_clustering_model, gaussian_mixture_model, dbscan_model, \
    birch_model
from feature_engineering.split_train_test import train_test_split
from load.load_sync_data import load_data_from_csv
from parser.check_create_directories import create_dir

KMEANS = "KMeans"
AGGLOMERATIVE = "Agglomerative_Clustering"
GAUSSIAN_MIXTURE_MODEL = "Gaussian_Mixture_Model"
DBSCAN = "DBSCAN"
BIRCH = "Birch"
SUPPORTED_MODELS = [KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH]


# ------------------------------------------------------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------------------------------------------------------- #

def feature_selector(dataset_path: str, n_iterations: int, clustering_model: str, output_path: str,
                     folder_name: str = "features_kmeans_plots"):
    # load dataset from csv file
    df = load_data_from_csv(dataset_path)

    # generate output path to save the plots
    output_path = create_dir(output_path, folder_name)

    # train test split
    train_set, test_set = train_test_split(df, 0.8, 0.2)

    # get the true (class) labels
    true_labels = train_set['class']

    # drop class and subclass column
    train_set = train_set.drop(['class', 'subclass'], axis=1)

    # remove class and subclass column and standardize features
    train_set = _standardize_features(train_set)

    # drop features with variance lower than variance_threshold
    train_set = _drop_low_variance_features(train_set, variance_threshold=0.1)
    print(f"Columns after dropping low variance features: {train_set.columns.tolist()}")

    feature_sets = []

    for i in range(1, n_iterations + 1):
        # Reset the best accuracy for each iteration
        best_ri = 0

        # Shuffle the column names at the beginning of each iteration
        shuffled_features = _shuffle_column_names(train_set)
        print(f"Shuffled Features (Iteration {i}): {shuffled_features}")

        # Reset the feature list for each iteration
        iter_feature_list = []

        # Temporary lists for storing the best features and metrics of the current iteration
        best_features = []
        accur_ri = []
        adj_rand_scores = []
        norm_mutual_infos = []

        # Cycle over the shuffled columns of the dataframe
        for feature in shuffled_features:
            # Add the feature to the feature list
            iter_feature_list.append(feature)

            # Get the corresponding columns
            features_train = train_set[iter_feature_list]

            if clustering_model == KMEANS:
                # kmeans feature_engineering
                labels = kmeans_model(features_train, n_clusters=3)

            elif clustering_model == AGGLOMERATIVE:
                # agglomerative feature_engineering
                labels = agglomerative_clustering_model(features_train, n_clusters=3)

            elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
                # gaussian mixture model
                labels = gaussian_mixture_model(features_train, n_components=3)

            elif clustering_model == DBSCAN:
                # DBSCAN feature_engineering
                labels = dbscan_model(features_train, 0.4, 10)

            elif clustering_model == BIRCH:
                # Birch feature_engineering
                labels = birch_model(features_train, n_clusters=3)

            else:
                raise ValueError(f"The model {clustering_model} is not supported. "
                                 f"Supported models are: {SUPPORTED_MODELS}")

            # Evaluate clustering with this feature set
            ri, ari, nmi = _evaluate_clustering(true_labels, labels)

            # if the Rand Index does not improve remove feature
            if ri <= best_ri:
                iter_feature_list.remove(feature)

            # if Rand Index improves add the feature to the feature list
            else:
                best_ri = ri
                best_features.append(iter_feature_list.copy())

                # Add results to the respective lists
                accur_ri.append(ri)
                adj_rand_scores.append(ari)
                norm_mutual_infos.append(nmi)

        print(f"Iteration {i}: Best Features - {best_features}")

        # X-axis will show the feature sets
        feature_names = ['\n'.join(features) for features in best_features]
        best_features_list = best_features[-1]

        print(best_features_list)

        if best_features_list not in feature_sets:

            # # generate the filename - name of the features used
            # best_features_str = '_'.join(best_features_list).replace(' ', '')
            # print(best_features_str)
            feature_sets.append(best_features_list)
            filename = f"{output_path}/feature_set{i}_results.png"

            # generate plot
            plt.figure(figsize=(14, 7))
            plt.plot(feature_names, accur_ri, marker='o', linestyle='-', color='#D36135', label='Rand Index')
            plt.plot(feature_names, adj_rand_scores, marker='x', linestyle='-', color='#386FA4',
                     label='Adjusted Rand Index')
            plt.plot(feature_names, norm_mutual_infos, marker='s', linestyle='-', color='#4D9078',
                     label='Normalized Mutual Info')
            plt.xlabel('Feature Sets')
            plt.ylabel('Scores')
            plt.title(clustering_model.replace('_', ' '))
            plt.xticks(rotation=0)  # Set rotation to 0 to keep the labels horizontal
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Plot saved with the following features: {best_features_list}")
        else:
            print("Feature set already saved")


# ------------------------------------------------------------------------------------------------------------------- #
# Private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _standardize_features(x: pd.DataFrame):
    # get new column names
    x_column_names = x.columns

    # standardize the features
    x = MinMaxScaler().fit_transform(x)

    # new dataframe without class and subclass column
    x = pd.DataFrame(x, columns=x_column_names)

    return x


def _evaluate_clustering(true_labels: pd.Series, predicted_labels: pd.Series):
    # calculate feature_engineering accuracy
    rand_index = np.round(rand_score(true_labels, predicted_labels), 2)

    # rand index adjusted for chance
    adjusted_rand_index = np.round(adjusted_rand_score(true_labels, predicted_labels), 2)

    # mutual information
    normalized_mutual_info = np.round(normalized_mutual_info_score(true_labels, predicted_labels), 2)

    return rand_index, adjusted_rand_index, normalized_mutual_info


def _drop_low_variance_features(x: pd.DataFrame, variance_threshold: float):
    # check low variance features
    var_thr = VarianceThreshold(threshold=variance_threshold)
    var_thr.fit(x)

    # low variance columns
    concol = [column for column in x.columns
              if column not in x.columns[var_thr.get_support()]]

    # drop columns
    x = x.drop(concol, axis=1)

    return x


def _shuffle_column_names(x: pd.DataFrame):
    column_names = list(x.columns)
    random.shuffle(column_names)
    return column_names
