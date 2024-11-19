# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from typing import List, Tuple
import os
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# internal imports
from constants import AGGLOMERATIVE
from .common import cluster_data, check_features, cluster_basic_data
import metrics
import load


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def imbalanced_clustering(main_path: str, sitting_perc: float, nr_windows: int, clustering_model: str, nr_clusters: int,
                          features_folder_name: str, feature_set: List[str], results_path: str):
    """
    This function generates imbalanced datasets and clusters. The results over all subjects are then saved in an Excel
    sheet.

    :param main_path: str
    Path to the main folder containing the dataset. For example:
    .../*main_folder*/subfolder/subsubfolder/dataset.csv
    .../*subjects_datasets*/subject_P001/phone_features_basic_activities/dataset.csv

    :param sitting_perc: float
    Percentage that the sitting activity must have (i.e., 0.5, 0.7, 0.9)

    :param nr_windows: int
    Number of windows/chunks to have of the underrepresented classes

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param features_folder_name: str
    Path to the folder containing the dataset. For example:
    .../main_folder/subfolder/*subsubfolder*/dataset.csv
    .../subjects_datasets/subject_P001/*phone_features_basic_activities*/dataset.csv

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :param results_path: str
    Path in which to save the Excel sheet with the results

    :return: None
    """
    # list for holding the results on each subject
    results = []

    # iterate through the subject folders
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)

        print(f"Testing subject: {subject_folder}")

        # iterate through the folders inside each subject folder
        for folder_name in os.listdir(subject_folder_path):

            print(os.listdir(subject_folder_path))
            # get the specified folder
            if folder_name == features_folder_name:

                # get the path to the dataset
                features_folder_path = os.path.join(subject_folder_path, features_folder_name)

                # check if there's only one csv file in the folder
                if len(os.listdir(features_folder_path)) == 1:
                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])

                    ari, std_ari = _cluster_imbalanced_basic_activities(dataset_path, sitting_perc, nr_windows,
                                                                        clustering_model, nr_clusters, feature_set)
                    # ari = cluster_basic_data(dataset_path, clustering_model, nr_clusters, feature_set)
                    results.append({
                        "Subject ID": subject_folder,
                        "ARI": ari,
                        "STD": std_ari
                    })
                    # Inform user
                    print(f"Clustering results for subject: {subject_folder}")
                    # print(f"Feature set used: {subject_feature_set_str}")
                    print(
                        f"Adjusted Rand Index: {ari}")

                else:
                    raise ValueError("Only one dataset per folder is allowed.")

    print("It ran")
    # Create DataFrame from results and save to Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(results_path, "please_Please.xlsx")
    results_df.to_excel(excel_path, index=False)

    print(f"Results saved to {excel_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _cluster_imbalanced_basic_activities(path: str, sitting_perc: float, nr_windows: int, clustering_model: str,
                                         nr_clusters: int, feature_set: List[str]) -> Tuple[float, float]:
    """
    This function applies a sliding window approach on a dataframe in order to extract multiple consecutive chunks of
    the standing still and walking medium data. These chunks are then added to the sitting data to form the final
    dataset. The clustering result is obtained  by doing the mean ARI over all datasets.

    :param path: str
    Path to the dataset (csv file)

    :param sitting_perc: float
    Percentage that the sitting activity must have (i.e., 0.5, 0.7, 0.9)

    :param nr_windows: int
    Number of windows/chunks to have of the underrepresented classes

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :return: Tuple[float, float]
    Mean ARI and standard deviation
    """
    # load dataframes from basic activities into a dictionary
    df_dict = load.load_basic_activities_only(path)

    # calculate chunk size based on len(sitting_df) and the sitting_percentage - divided by 2 for walking and standing
    chunk_size = int((len(df_dict['sitting']) / sitting_perc - len(df_dict['sitting'])) // 2)

    print(f"chunk_size: {chunk_size}")

    # get starts and stops for walking
    chunk_indices_walking = _sliding_window(df_dict['walking'], chunk_size, nr_windows)

    # get start and stops for standing
    chunk_indices_standing = _sliding_window(df_dict['standing'], chunk_size, nr_windows)

    list_walking_chunks = []
    list_standing_chunks = []

    # cut the chunks and get the lists of chunks - walking
    for start_w, stop_w in chunk_indices_walking:
        temp_df = df_dict['walking'].iloc[start_w: stop_w]
        list_walking_chunks.append(temp_df)

    print(f"Number of walking chunks: {len(list_walking_chunks)}")

    # cut the chunks and get the lists of chunks - standing
    for start_s, stop_s in chunk_indices_standing:
        temp_df = df_dict['standing'].iloc[start_s: stop_s]
        list_standing_chunks.append(temp_df)

    print(f"Number of standing chunks: {len(list_standing_chunks)}")

    list_ari = []

    # initialize chunk counter
    chunk_counter = 0

    # build different datasets with the different chunks
    for walking_chunk, standing_chunk in zip(list_walking_chunks, list_standing_chunks):

        chunk_counter += 1

        # Reset temp_list for each new dataset combination
        temp_list = [df_dict['sitting'], walking_chunk, standing_chunk]

        temp_dataset = pd.concat(temp_list, ignore_index=True)

        true_labels = temp_dataset['class']

        print(f"Class counts for {sitting_perc} % sitting: {temp_dataset['subclass'].value_counts()}")

        # check if the feature set exists in the columns
        check_features(temp_dataset, feature_set)

        # get only the feature set
        temp_dataset = temp_dataset[feature_set]

        # Initialize scaler
        scaler = MinMaxScaler()

        # scale the train set
        temp_dataset = scaler.fit_transform(temp_dataset)

        # put the train and test sets back into a pandas dataframe
        temp_dataset = pd.DataFrame(temp_dataset, columns=feature_set)

        # cluster
        if clustering_model == AGGLOMERATIVE:
            cluster_labels = cluster_data(clustering_model, temp_dataset, nr_clusters)
        else:
            cluster_labels = cluster_data(clustering_model, temp_dataset, nr_clusters, temp_dataset)

        # evaluate
        _, ari, _ = metrics.evaluate_clustering(true_labels, cluster_labels)

        list_ari.append(ari)

    mean_ari = np.round(np.mean(list_ari), 4)
    std_ari = np.round(np.std(list_ari), 4)
    print(f"mean ari: {mean_ari}")

    return mean_ari, std_ari


def _sliding_window(df: pd.DataFrame, window_size: int, num_windows: int) -> List[Tuple[int, int]]:
    """
    Perform a sliding window approach over a pandas DataFrame.

    :param df: pd.DataFrame
    The DataFrame over which the sliding window is applied.

    :param window_size: int
    The size of each window.

    :param num_windows: int
    The number of windows to extract.

    :return: List[Tuple[int, int]]
    A list containing the start and stop indices of the corresponding windows.
    Each list element is a tuple pair of the start index and the stop index.
    """
    # Total number of rows in the DataFrame
    num_rows = len(df)

    # Calculate the total possible windows without any overlap
    max_possible_windows = num_rows - window_size + 1

    if max_possible_windows <= 0:
        raise ValueError("Window size is larger than the DataFrame length.")

    # Calculate the necessary overlap
    if num_windows > max_possible_windows:
        raise ValueError("Number of windows requested exceeds the possible number of windows.")

    # Calculate the step size (distance between the start of each window)
    step_size = (num_rows - window_size) / (num_windows - 1) if num_windows > 1 else num_rows

    print(f"step size: {step_size}")

    # Generate the start indices for the windows
    start_indices = [round(step_size * i) for i in range(num_windows)]

    # Create the list of windows
    windows = [(start, start + window_size) for start in start_indices]

    # If the number of windows exceeds the requested number, randomly select the necessary amount
    if len(windows) > num_windows:
        windows = random.sample(windows, num_windows)

    return windows

# def extract_random_chunks(dataframe, chunk_size, num_chunks):
#     """
#     Extracts random, non-overlapping chunks from a DataFrame using a sliding window approach.
#
#     Parameters:
#     - dataframe (pd.DataFrame): The DataFrame from which to extract chunks.
#     - chunk_size (int): The size of each chunk.
#     - num_chunks (int): The number of unique chunks to extract.
#
#     Returns:
#     - list of tuples: Each tuple contains the start and end index of the chunks selected.
#     """
#     num_rows = len(dataframe)
#
#     # Calculate the maximum start index for a chunk
#     max_start_index = num_rows - chunk_size + 1
#
#     if max_start_index < 1:
#         raise ValueError("Chunk size is larger than the dataframe.")
#
#     # Generate all possible start indices for the chunks
#     all_start_indices = np.arange(max_start_index)
#
#     # Shuffle the indices to randomly select them without replacement
#     np.random.shuffle(all_start_indices)
#
#     # Select the first num_chunks indices from the shuffled indices (if enough are available)
#     if num_chunks > len(all_start_indices):
#         raise ValueError("The number of requested chunks exceeds the number of available unique chunks.")
#
#     selected_indices = all_start_indices[:num_chunks]
#
#     # Create a list of tuples containing the start and end indices for each chunk
#     chunk_indices = [(start, start + chunk_size) for start in selected_indices]
#
#     return chunk_indices
