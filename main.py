from feature_engineering.feature_selection import feature_selector
from feature_extraction.feature_extraction import feature_extractor, generate_cfg_file
from processing.processor import processing
from synchronization.synchronization import synchronization
from visualization.visualize_filters import visualize_filtering_results
from visualization.visualize_sync import visualize_sync_signals


def main():
    # Set these booleans to True or False depending on which steps you want to run
    do_synchronization = False
    do_processing = False
    do_generate_cfg_file = False
    do_feature_extraction = False
    do_feature_selection = False

    if do_synchronization:
        raw_data_in_path = "D:/tese_backups/raw_signals_backups/acquisitions/P004"
        sync_android_out_path = ("C:/Users/srale/OneDrive - FCT "
                                 "NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_android_P004")
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/synchronized_P004"
        selected_sensors = {'phone': ['acc', 'gyr', 'mag']}
        sync_type = 'crosscorr'
        evaluation_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_evaluation_P004"
        synchronization(raw_data_in_path, sync_android_out_path, selected_sensors, output_path, sync_type,
                        evaluation_path)

    if do_processing:
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
        filtered_folder_name = "filtered_tasks_P004"
        raw_folder_name = "raw_tasks_P004"
        sync_data_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/sync_android_P004"
        processing(sync_data_path, output_path, raw_folder_name, filtered_folder_name)

    if do_generate_cfg_file:
        cfg_path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
        generate_cfg_file(cfg_path)

    if do_feature_extraction:
        subclasses = ['standing_still', 'walk_medium', 'sit']
        main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone/filtered_tasks_P004"
        output_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
        feature_extractor(main_path, output_path, subclasses)

    if do_feature_selection:
        dataset_path = ("C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P003/acc_gyr_mag_phone/features_basic_activities"
                        "/acc_gyr_mag_phone_features_P003.csv")
        output_path_plots = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P003/acc_gyr_mag_phone"
        clustering_model = "KMeans"
        nr_iterations = 20
        feature_sets, best_acc = feature_selector(dataset_path, nr_iterations, clustering_model, output_path_plots)
        print(feature_sets)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from collections import Counter
from itertools import combinations
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import rand_score, adjusted_rand_score, normalized_mutual_info_score

from feature_engineering.models import kmeans_model, agglomerative_clustering_model, gaussian_mixture_model, \
    dbscan_model, birch_model
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
                     folder_name: str = "features_kmeans_plots", save_plots: bool = False):
    # load dataset from csv file
    df = load_data_from_csv(dataset_path)

    # generate output path to save the plots
    output_path = create_dir(output_path, folder_name)

    # train test split
    train_set, _ = train_test_split(df, 0.8, 0.2)

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
    feature_sets_accur = []

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
        highest_accuracy = accur_ri[-1]

        print(best_features_list)

        if best_features_list not in feature_sets:

            feature_sets.append(best_features_list)
            feature_sets_accur.append(highest_accuracy)
            file_path = f"{output_path}/feature_set{i}_results.png"

            # save plots if save_plots == True
            if save_plots:
                # plot clustering results for that feature set
                _save_plot_clustering_results(feature_names, accur_ri, adj_rand_scores, norm_mutual_infos, file_path,
                                              clustering_model)
                # inform user
                print(f"Plot saved with the following features: {best_features_list}")
        else:
            # inform user
            print(f"Feature combination already tested.")

    return feature_sets, feature_sets_accur

from tqdm import tqdm


def get_all_subjects_best_features(main_path: str, features_folder_name: str, n_iterations, clustering_model,
                                   output_path):
    all_best_features = []
    all_best_accuracies = []

    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)
        print(f"Selecting best features for subject: {subject_folder}")

        for sub_folder in os.listdir(subject_folder_path):
            sub_folder_path = os.path.join(subject_folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):

                features_folder_path = os.path.join(sub_folder_path, features_folder_name)
                if os.path.isdir(features_folder_path):
                    feature_files = os.listdir(features_folder_path)

                    if len(feature_files) == 1:
                        dataset_path = os.path.join(features_folder_path, feature_files[0])

                        # Get the best feature sets for the subject
                        feature_sets, accuracies = feature_selector(dataset_path, n_iterations, clustering_model, output_path,
                                                                     save_plots=True)

                        # Filter for the best feature sets and their accuracies
                        best_feature_sets, best_acc = filter_best_feature_sets(feature_sets, accuracies)
                        all_best_features.append(best_feature_sets)
                        all_best_accuracies.append(best_acc)

                        print("#########################################################################")
                        print(f"SUBJECT: {subject_folder}")
                        print("#########################################################################")
                        print(f"Feature sets with the highest accuracies:")
                        for feat, acc in zip(best_feature_sets, best_acc):
                            print(f"Features: {feat} \n Rand Index: {acc}")

                    else:
                        raise ValueError(f"Too many files: {len(feature_files)} files. Only one dataset per folder.")
                else:
                    print(f"No features folder found in: {sub_folder_path}")

    return all_best_features, all_best_accuracies

# ------------------------------------------------------------------------------------------------------------------- #
# Private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _save_plot_clustering_results(feature_names: List[str], accur_ri: List[float], adj_rand_scores: List[float],
                                  norm_mutual_infos: List[float], file_path: str, clustering_model: str) -> None:
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
    plt.savefig(file_path)
    plt.close()


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


def _remove_first_letter(feature_sets: List[List[str]]) -> List[List[str]]:
    return [[feature[1:] for feature in feature_set] for feature_set in feature_sets]


def _get_most_common_pair(feature_sets: List[List[str]]) -> List[str]:
    # Filter out the feature sets that contain exactly two features
    two_feature_sets = [frozenset(feature_set) for feature_set in feature_sets if len(feature_set) == 2]

    most_common_pair = Counter(two_feature_sets).most_common(1)

    # Return the most common pair as a list
    if most_common_pair:
        return list(most_common_pair[0][0])
    return []


def _get_most_common_feature_set(feature_sets: List[List[str]]) -> List[str]:
    # Find the most common pair among sets with exactly two features
    most_common_pair = _get_most_common_pair(feature_sets)
    if not most_common_pair:
        return []

    # Count how often this pair appears in all feature sets
    pair_count = 0
    pair_set = frozenset(most_common_pair)
    for feature_set in feature_sets:
        if pair_set.issubset(feature_set):
            pair_count += 1

    return most_common_pair if pair_count > 0 else []


# def get_all_subjects_feature_sets(main_path: str, features_folder_name: str, n_iterations, clustering_model,
#                                   output_path):
#     # List to hold the best feature sets of each subject
#     all_subject_feature_sets = []
#
#     for subject_folder in os.listdir(main_path):
#         subject_folder_path = os.path.join(main_path, subject_folder)
#         print(f"Processing subject folder: {subject_folder_path}")
#
#         for sub_folder in os.listdir(subject_folder_path):
#             sub_folder_path = os.path.join(subject_folder_path, sub_folder)
#             if os.path.isdir(sub_folder_path):
#                 print(f"Processing sub-folder: {sub_folder_path}")
#
#                 features_folder_path = os.path.join(sub_folder_path, features_folder_name)
#                 if os.path.isdir(features_folder_path):
#                     print(f"Processing features folder: {features_folder_path}")
#                     feature_files = os.listdir(features_folder_path)
#                     print(f"Found feature files: {feature_files}")
#
#                     if len(feature_files) == 1:
#                         dataset_path = os.path.join(features_folder_path, feature_files[0])
#                         print(f"Processing dataset file: {dataset_path}")
#                         # Get the best feature sets for the subject
#                         feature_sets = feature_selector(dataset_path, n_iterations, clustering_model, output_path, save_plots=False)
#                         print(f"Feature sets for {dataset_path}: {feature_sets}")
#                         all_subject_feature_sets.extend(feature_sets)
#                     else:
#                         raise ValueError(f"Too many files: {len(feature_files)} files. Only one dataset per folder.")
#                 else:
#                     print(f"No features folder found in: {sub_folder_path}")
#
#     print(f"All subject feature sets: {all_subject_feature_sets}")
#     return all_subject_feature_sets
#
#
# def count_feature_frequency(feature_sets):
#     feature_counter = Counter()
#     for feature_set in feature_sets:
#         feature_counter.update(feature_set)
#     return feature_counter
#
# def get_most_common_features(feature_counter, top_n=10):
#     most_common_features = feature_counter.most_common(top_n)
#     return [feature for feature, count in most_common_features]
def filter_best_feature_sets(feature_sets, feature_sets_accur):
    # Find the highest accuracy
    highest_accuracy = max(feature_sets_accur)

    # Filter feature sets and accuracies that have the highest accuracy
    best_feature_sets = [feature_sets[i] for i, accur in enumerate(feature_sets_accur) if accur == highest_accuracy]
    best_accuracies = [accur for accur in feature_sets_accur if accur == highest_accuracy]

    return best_feature_sets, best_accuracies

main_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects" #/P004/acc_gyr_mag_phone/features_basic_activities/acc_gyr_mag_phone_features_P004.csv"
output_path_plots = "C:/Users/srale/OneDrive - FCT NOVA/Tese/subjects/P004/acc_gyr_mag_phone"
clustering_model = "KMeans"
nr_iterations = 20

#
# all_subject_feature_sets = get_all_subjects_feature_sets(dataset_path, "features_basic_activities", nr_iterations, clustering_model, output_path_plots)
# print("all feature sets", all_subject_feature_sets)
# # Remove the first letter from each feature in the feature sets
# all_subject_feature_sets = _remove_first_letter(all_subject_feature_sets)
#
# # Count feature frequency
# feature_counter = count_feature_frequency(all_subject_feature_sets)
#
# # Get the most common features
# most_common_features = get_most_common_features(feature_counter, top_n=10)
# #
# print("Most Common Features Across All Subjects:", most_common_features)


# all_subject_best_features = get_all_subjects_best_features(dataset_path, "features_basic_activities", nr_iterations, clustering_model, output_path_plots)
# print("All Subject Best Features:", all_subject_best_features)


# Filter for the best feature sets and their accuracies
all_best_features, all_best_accuracies = get_all_subjects_best_features(main_path, "features_basic_activities", nr_iterations, clustering_model, output_path_plots)



def aggregate_best_feature_sets(all_best_features):
    # Flatten the list of lists of lists into a single list of lists
    all_feature_sets = [feature_set for subject_feature_sets in all_best_features for feature_set in subject_feature_sets]
    return all_feature_sets

def count_feature_frequency(feature_sets):
    # Count the frequency of each feature
    feature_counter = Counter(feature for feature_set in feature_sets for feature in feature_set)
    return feature_counter

def select_top_features(feature_counter, top_n=10):
    # Select the top N features based on their frequency
    most_common_features = [feature for feature, count in feature_counter.most_common(top_n)]
    return most_common_features

# Step 1: Aggregate Best Feature Sets
all_feature_sets = aggregate_best_feature_sets(all_best_features)
print("All Feature Sets:", all_feature_sets)

# Step 2: Count Feature Frequency
feature_counter = count_feature_frequency(all_feature_sets)
print("Feature Frequencies:", feature_counter)

# Step 3: Select Top Features
top_n = 10  # Number of top features to select
most_common_features = select_top_features(feature_counter, top_n)
print("Most Common Features:", most_common_features)
