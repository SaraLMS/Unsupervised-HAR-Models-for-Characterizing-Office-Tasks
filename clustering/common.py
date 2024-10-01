# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib.colors import to_hex, LinearSegmentedColormap
import re
import os
from typing import List, Tuple
import numpy as np
import load
import metrics
import models
from constants import KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, SUPPORTED_MODELS, CLASS, SUBCLASS

# set text font to palatino
plt.style.use('my_style')


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def cluster_basic_data(dataset_path: str, clustering_model: str, nr_clusters: int, feature_set: List[str]) -> float:
    """
    Clusters basic activities, 'sitting', 'standing', and 'walking_medium' for experiment 3. For this experiment, all
    100 % of the available data is used for clustering. The feature set used is provided by the user (feature_set) and
    features are scaled using sklearn min_max_scaler.

    :param dataset_path: str
    Path to the dataset (csv file) with the columns as features and rows as instances.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :return: float
    The adjusted rand index (ari) for this clustering.

    """
    # load dataset
    dict_dfs = load.load_basic_activities_only(dataset_path)

    # get the dataframes from the 3 basic activities
    list_dfs = [dict_dfs['sitting'], dict_dfs['standing'], dict_dfs['walking']]

    # concat to a dataframe
    df = pd.concat(list_dfs, ignore_index=True)

    # check if the feature set exists in the columns
    check_features(df, feature_set)

    # get the true labels
    true_labels = df[CLASS]

    # get only the set of features
    df = df[feature_set]

    # Initialize scaler
    scaler = MinMaxScaler()

    # scale the train set
    df = scaler.fit_transform(df)

    # put the train and test sets back into a pandas dataframe
    df = pd.DataFrame(df, columns=feature_set)

    # Cluster data
    if clustering_model == AGGLOMERATIVE:
        pred_labels = cluster_data(clustering_model, df, nr_clusters)
    else:
        pred_labels = cluster_data(clustering_model, df, nr_clusters, df)

    # Evaluate clustering
    _, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    return ari


def cluster_data(clustering_model: str, train_set: pd.DataFrame, n_clusters: int, test_set: pd.DataFrame = None) -> pd.Series:
    """
    This function does the following:
    (1) clustering on the train_set.
    (2) test_set instances are then appointed to the preformed clusters.

    If only step (1) is needed, pass the same dataframe to the train_set and test_set parameters.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param train_set: pd.DataFrame
    pandas DataFrame containing the training data. Columns are features and rows are data instances, with the column
    names being the feature names.

    :param test_set: pd.DataFrame
    pandas DataFrame containing the testing data. Columns are features and rows are data instances, with the column
    names being the feature names.

    :param n_clusters: int
    Number of clusters to find

    :return: pd.Series
    A pandas series containing the predicted labels
    """
    # Perform clustering based on the selected model
    if clustering_model == KMEANS:
        labels = models.kmeans_model(train_set, test_set, n_clusters)

    # if AGG clustering, the model takes only the train set, as it does not allow for predicting new data
    elif clustering_model == AGGLOMERATIVE:
        labels = models.agglomerative_clustering_model(train_set, n_clusters)

    elif clustering_model == GAUSSIAN_MIXTURE_MODEL:
        labels = models.gaussian_mixture_model(train_set, test_set, n_clusters)
    else:
        raise ValueError(f"The model {clustering_model} is not supported. "
                         f"Supported models are: {SUPPORTED_MODELS}")
    return labels


def normalize_subclass(subclass: str) -> str:
    """
    Remove numeric indices from the subclass names (i.e., 'stand_still1' to 'stand_still'). Remove the 'up' and 'down'
    form 'stairs' instances. This function is used for the confusion matrix.

    :param subclass: str
    A str pertaining to the subclass names

    :return: pd.Series
    Modified subclass name
    """
    # Normalize all variations of 'stairs' to 'stairs'
    if subclass.startswith('stairs'):
        return 'stairs'
    # Remove numeric suffixes from other subclass names
    return re.sub(r'\d+', '', subclass)


def format_subclass_name(subclass_name: str) -> str:
    """
    Removes underscores and changes names to uniformize all subclass names

    :param subclass_name: str
    A string pertaining to the subclass name

    :return: str
    Modified subclass name

    """
    # Replace underscores with spaces and handle specific cases
    formatted_name = subclass_name.replace('_', ' ')
    if formatted_name == 'standing still':
        formatted_name = 'stand still'
    if formatted_name == 'standing gestures':
        formatted_name = 'stand gestures'
    if formatted_name == 'standing coffee':
        formatted_name = 'stand coffee'
    if formatted_name == 'standing folders':
        formatted_name = 'stand folders'
    return formatted_name


def cluster_subject_all_activities(dataset_path: str, clustering_model: str, nr_clusters: int, feature_set: List[str],
                                   subject_id: str, plots_path: str = None, save_confusion_matrix: bool = True,
                                   train_size: float = 0.8, test_size: float = 0.2) -> Tuple[float, float]:
    """
    This function does the following:
    (1) clustering on the train_set.
    (2) test_set instances are then appointed to the preformed clusters.
    If only step (1) is needed, pass the same dataframe to the train_set and test_set parameters.

    This function also generates a confusion matrix that can be saved to plots_path if save_confusion_matrix is set
    to True. This function can only be used for 'all' activities since the confusion matrices are build specifically
    for it. 'all' activities are: 'sitting', 'standing_still', 'walking_medium', 'standing_gestures', 'stairs',
    'walk_fast', 'walk_slow', 'coffee', 'folders'.

    :param dataset_path: str
    Path to the dataset (csv file) with the columns as features and rows as instances.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :param subject_id: str
    Identification fo the subject or of the dataset. Used for the confusion matrix only

    :param save_confusion_matrix: bool (default = True)
    Save the confusion matrix as PNG. False not to save.

    :param plots_path: str
    Path to the location in which to store the confusion matrix

    :param train_size: float
    Train size ratio, between 0 and 1

    :param test_size: float
    Test size ration, between 0 and 1

    :return: Tuple[float, float]
    The adjusted rand index and normalized mutual information results.
    """
    # Train-test split
    train_set, test_set = load.train_test_split(dataset_path, train_size, test_size)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(train_set, feature_set)
    check_features(test_set, feature_set)

    # Get true labels for evaluation
    true_labels = test_set[CLASS]

    # Save the subclass
    subclass_column = test_set[SUBCLASS]

    # Normalize the subclass names
    normalized_subclass_column = subclass_column.apply(normalize_subclass)

    # Get only the wanted features in the train and test sets
    train_set = train_set[feature_set]
    test_set = test_set[feature_set]

    # Initialize scaler
    scaler = MinMaxScaler()

    # scale the train set
    train_set = scaler.fit_transform(train_set)

    # scale the test set with the same normalization parameters as the train set
    test_set = scaler.transform(test_set)

    # put the train and test sets back into a pandas dataframe
    train_set = pd.DataFrame(train_set, columns=feature_set)
    test_set = pd.DataFrame(test_set, columns=feature_set)

    # Cluster data
    if clustering_model == AGGLOMERATIVE:
        pred_labels = cluster_data(clustering_model, test_set, nr_clusters)

    else:
        pred_labels = cluster_data(clustering_model, train_set, nr_clusters, test_set)

    # Evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    ############### BUILD CONFUSION MATRIX #######################

    # Results dataframe
    results_df = pd.DataFrame(
        {'class': true_labels, 'subclass': subclass_column, 'normalized_subclass': normalized_subclass_column,
         'predicted_labels': pred_labels})

    # Order subclasses
    subclass_order = ['sit', 'standing_still', 'standing_gestures', 'standing_coffee', 'standing_folders',
                      'walk_slow', 'walk_medium', 'walk_fast', 'stairs']

    # Create a confusion matrix of normalized subclasses and clusters
    subclass_cluster_matrix = pd.crosstab(results_df['normalized_subclass'], results_df['predicted_labels'])
    subclass_cluster_matrix = subclass_cluster_matrix.reindex(subclass_order)

    # Define base colors for the rows based on the class
    base_colors = {
        'sit': '#E59B23',
        'standing': '#81B29A',
        'walking': '#E07A5F'
    }

    # Create a mapping from subclasses to their base colors
    subclass_base_color = {
        'sit': base_colors['sit'],
        'standing_still': base_colors['standing'],
        'standing_gestures': base_colors['standing'],
        'standing_coffee': base_colors['standing'],
        'standing_folders': base_colors['standing'],
        'walk_slow': base_colors['walking'],
        'walk_medium': base_colors['walking'],
        'walk_fast': base_colors['walking'],
        'stairs': base_colors['walking']
    }

    max_value = subclass_cluster_matrix.values.max()

    # Create a color matrix based on the values and their base colors
    color_matrix = subclass_cluster_matrix.copy()
    for row in color_matrix.index:
        for col in color_matrix.columns:
            value = color_matrix.loc[row, col]
            intensity = value / max_value if max_value > 0 else 0
            # Adjust intensity: higher value -> darker color
            color_matrix.loc[row, col] = to_hex(
                LinearSegmentedColormap.from_list('custom_colormap', ['#ffffff', subclass_base_color[row]])(intensity))

    # Format subclass names for display
    formatted_subclass_order = [format_subclass_name(name) for name in subclass_order]

    # Update the subclass_cluster_matrix to use formatted names
    subclass_cluster_matrix.index = subclass_cluster_matrix.index.map(format_subclass_name)
    subclass_cluster_matrix = subclass_cluster_matrix.reindex(formatted_subclass_order)

    # Plot the confusion matrix with custom colors
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(subclass_cluster_matrix, annot=True, fmt='g', cbar=False,
                     yticklabels=formatted_subclass_order, linewidths=.5, linecolor='black', cmap='Greys',
                     mask=subclass_cluster_matrix.isna(), annot_kws={"weight": "bold", "fontsize": 16})
    title = f"Subject {subject_id}"
    plt.title(title, fontsize=18)
    plt.xlabel('Cluster Labels', fontsize=18)
    plt.ylabel('Subclass Labels', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Color the background of each cell
    for i in range(subclass_cluster_matrix.shape[0]):
        for j in range(subclass_cluster_matrix.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix.iloc[i, j], edgecolor='black'))

    # Add horizontal and vertical lines to separate rows and columns
    ax.hlines(np.arange(1, len(subclass_order)), *ax.get_xlim(), color='black', linewidth=1)
    ax.vlines(np.arange(1, subclass_cluster_matrix.shape[1]), *ax.get_ylim(), color='black', linewidth=1)

    # Save the plot with high resolution if save_path is provided
    if save_confusion_matrix and plots_path is not None:
        save_path = os.path.join(plots_path, subject_id + '.png')
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')

    plt.show()

    return ari, nmi


def check_features(dataframe, feature_set):
    missing_features = [feature for feature in feature_set if feature not in dataframe.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the dataframe columns: {missing_features}")


def cluster_subject_basic_matrix(dataset_path: str, clustering_model: str, nr_clusters: int, feature_set: List[str],
                                 subject_id: str, save_path: str = None, save_confusion_matrix: bool = True,
                                 train_size: float = 0.8, test_size: float = 0.2) -> Tuple[float, float]:
    """
    This function does the following:
    (1) clustering on the train_set.
    (2) test_set instances are then appointed to the preformed clusters.
    If only step (1) is needed, pass the same dataframe to the train_set and test_set parameters.

    This function also generates a confusion matrix that can be saved to plots_path if save_confusion_matrix is set
    to True. This function can only be used for 'basic' activities since the confusion matrices are build specifically
    for it. 'basic' activities are: 'sitting', 'standing_still', and 'walking_medium'.

    :param dataset_path: str
    Path to the dataset (csv file) with the columns as features and rows as instances.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :param subject_id: str
    Identification fo the subject or of the dataset. Used for the confusion matrix only

    :param save_confusion_matrix: bool (default = True)
    Save the confusion matrix as PNG. False not to save.

    :param save_path: str
    Path to the location in which to store the confusion matrix

    :param train_size: float
    Train size ratio, between 0 and 1

    :param test_size: float
    Test size ration, between 0 and 1

    :return: Tuple[float, float]
    The adjusted rand index and normalized mutual information results.
    """
    # Train-test split
    train_set, test_set = load.train_test_split(dataset_path, train_size, test_size)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(train_set, feature_set)
    check_features(test_set, feature_set)

    # Get true labels for evaluation
    true_labels = test_set[CLASS]

    # Save the subclass
    subclass_column = test_set[SUBCLASS]

    # Normalize the subclass names
    normalized_subclass_column = subclass_column.apply(normalize_subclass)

    # Get only the wanted features in the train and test sets
    train_set = train_set[feature_set]
    test_set = test_set[feature_set]

    # Initialize scaler
    scaler = MinMaxScaler()

    # scale the train set
    train_set = scaler.fit_transform(train_set)

    # scale the test set with the same normalization parameters as the train set
    test_set = scaler.transform(test_set)

    # put the train and test sets back into a pandas dataframe
    train_set = pd.DataFrame(train_set, columns=feature_set)
    test_set = pd.DataFrame(test_set, columns=feature_set)

    # Cluster data
    if clustering_model == AGGLOMERATIVE:
        pred_labels = cluster_data(clustering_model, test_set, nr_clusters)

    else:
        pred_labels = cluster_data(clustering_model, train_set, nr_clusters, test_set)

    # Evaluate clustering
    _, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    ############### BUILD CONFUSION MATRIX #######################

    # Results dataframe
    results_df = pd.DataFrame(
        {'class': true_labels, 'subclass': subclass_column, 'normalized_subclass': normalized_subclass_column,
         'predicted_labels': pred_labels})

    # Order subclasses
    subclass_order = ['sit', 'standing_still', 'walk_medium']

    # Create a confusion matrix of normalized subclasses and clusters
    subclass_cluster_matrix = pd.crosstab(results_df['normalized_subclass'], results_df['predicted_labels'])
    subclass_cluster_matrix = subclass_cluster_matrix.reindex(subclass_order)

    # Define base colors for the rows based on the class
    base_colors = {
        'sit': '#E59B23',
        'standing': '#81B29A',
        'walking': '#E07A5F'
    }

    # Create a mapping from subclasses to their base colors
    subclass_base_color = {
        'sit': base_colors['sit'],
        'standing_still': base_colors['standing'],
        'walk_medium': base_colors['walking'],
    }

    max_value = subclass_cluster_matrix.values.max()

    # Create a color matrix based on the values and their base colors
    color_matrix = subclass_cluster_matrix.copy()
    for row in color_matrix.index:
        for col in color_matrix.columns:
            value = color_matrix.loc[row, col]
            intensity = value / max_value if max_value > 0 else 0
            # Adjust intensity: higher value -> darker color
            color_matrix.loc[row, col] = to_hex(
                LinearSegmentedColormap.from_list('custom_colormap', ['#ffffff', subclass_base_color[row]])(intensity))

    # Format subclass names for display
    formatted_subclass_order = [format_subclass_name(name) for name in subclass_order]

    # Update the subclass_cluster_matrix to use formatted names
    subclass_cluster_matrix.index = subclass_cluster_matrix.index.map(format_subclass_name)
    subclass_cluster_matrix = subclass_cluster_matrix.reindex(formatted_subclass_order)

    # Plot the confusion matrix with custom colors
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(subclass_cluster_matrix, annot=True, fmt='g', cbar=False,
                     yticklabels=formatted_subclass_order, linewidths=.5, linecolor='black', cmap='Greys',
                     mask=subclass_cluster_matrix.isna(),
                     annot_kws={"weight": "bold", "fontsize": 18})  # Make numbers bold
    title = f"Subject {subject_id}"
    plt.title(title, fontsize=18)
    plt.xlabel('Cluster Labels', fontsize=18)
    plt.ylabel('Subclass Labels', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=16)

    # Color the background of each cell
    for i in range(subclass_cluster_matrix.shape[0]):
        for j in range(subclass_cluster_matrix.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix.iloc[i, j], edgecolor='black'))

    # Add horizontal and vertical lines to separate rows and columns
    ax.hlines(np.arange(1, len(subclass_order)), *ax.get_xlim(), color='black', linewidth=1)
    ax.vlines(np.arange(1, subclass_cluster_matrix.shape[1]), *ax.get_ylim(), color='black', linewidth=1)

    # Save the plot with high resolution if save_path is provided
    if save_confusion_matrix and save_path is not None:
        save_path = os.path.join(save_path, subject_id + '.png')
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')

    plt.show()

    return ari, nmi




