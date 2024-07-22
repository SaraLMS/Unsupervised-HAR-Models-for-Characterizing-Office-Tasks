# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from matplotlib.colors import to_rgba, to_hex, LinearSegmentedColormap
import re
import os
from typing import List
import numpy as np
import load
import metrics
import models
from constants import KMEANS, AGGLOMERATIVE, GAUSSIAN_MIXTURE_MODEL, DBSCAN, BIRCH, SUPPORTED_MODELS, CLASS, SUBCLASS


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
        labels = models.agglomerative_clustering_model(train_set, n_clusters)
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

def normalize_subclass(subclass):
    # Normalize all variations of 'stairs' to 'stairs'
    if subclass.startswith('stairs'):
        return 'stairs'
    # Remove numeric suffixes from other subclass names
    return re.sub(r'\d+', '', subclass)


def cluster_subject(dataset_path: str, clustering_model: str, feature_set: List[str], subject_id: str, train_size: float = 0.8,
                    test_size: float = 0.2):
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

    # Normalize the features
    train_set = normalize_features(train_set)
    test_set = normalize_features(test_set)

    # Cluster data
    pred_labels = cluster_data(clustering_model, train_set, test_set, n_clusters=3)

    # Evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    # Results dataframe
    results_df = pd.DataFrame({'class': true_labels, 'subclass': subclass_column, 'normalized_subclass': normalized_subclass_column, 'predicted_labels': pred_labels})

    # Count how many points from each class are inside each cluster
    cluster_counts = results_df.groupby('predicted_labels')['class'].value_counts().unstack(fill_value=0)

    # Determine the dominant class for each cluster
    cluster_dominant_class = cluster_counts.idxmax(axis=1)

    # Create a mapping from predicted labels to dominant class labels
    label_mapping = {cluster: dominant_class for cluster, dominant_class in cluster_dominant_class.items()}

    # Map predicted labels to dominant class labels
    results_df['predicted_labels'] = results_df['predicted_labels'].map(label_mapping)

    # Update class names
    class_mapping = {1: 'standing', 2: 'sitting', 3: 'walking'}
    results_df['predicted_labels'] = results_df['predicted_labels'].map(class_mapping)

    # Order subclasses
    subclass_order = ['sit', 'standing_still', 'standing_gestures', 'standing_coffee', 'standing_folders',
                      'walk_slow', 'walk_medium', 'walk_fast', 'stairs']

    # Create a confusion matrix of normalized subclasses and clusters
    subclass_cluster_matrix = pd.crosstab(results_df['normalized_subclass'], results_df['predicted_labels'])
    subclass_cluster_matrix = subclass_cluster_matrix.reindex(subclass_order)

    ############### BUILD HEAT MAP #######################
    # Define base colors for the rows based on the class
    base_colors = {
        'sit': '#D36336',
        'standing': '#2F5D89',
        'walking': '#396A59'
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

    # Plot the confusion matrix with custom colors
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(subclass_cluster_matrix, annot=True, fmt='g', cbar=False,
                     yticklabels=subclass_order, linewidths=.5, linecolor='black', cmap='Greys',
                     mask=subclass_cluster_matrix.isna())
    title = f"COnfusion_matrix: subject {subject_id}"
    plt.title(title)
    plt.xlabel('Cluster Labels')
    plt.ylabel('Subclass Labels')

    # Color the background of each cell
    for i in range(subclass_cluster_matrix.shape[0]):
        for j in range(subclass_cluster_matrix.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix.iloc[i, j], edgecolor='black'))

    # Add horizontal and vertical lines to separate rows and columns
    ax.hlines(np.arange(1, len(subclass_order)), *ax.get_xlim(), color='black', linewidth=1)
    ax.vlines(np.arange(1, subclass_cluster_matrix.shape[1]), *ax.get_ylim(), color='black', linewidth=1)

    # # Save the plot with high resolution if save_path is provided
    # if plots_path:
    #     save_path = os.path.join(plots_path, subject_id + '.png')
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #
    # plt.show()

    return ari, nmi





def check_features(dataframe, feature_set):
    missing_features = [feature for feature in feature_set if feature not in dataframe.columns]
    if missing_features:
        raise ValueError(f"The following features are missing from the dataframe columns: {missing_features}")


def cluster_subject_basic_matrix(dataset_path: str, clustering_model: str, feature_set: List[str], subject_id: str,
                    save_path: str = None, train_size: float = 0.8,
                    test_size: float = 0.2):
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

    # Normalize the features
    train_set = normalize_features(train_set)
    test_set = normalize_features(test_set)

    # Cluster data
    pred_labels = cluster_data(clustering_model, train_set, test_set, n_clusters=3)

    # Evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    # Results dataframe
    results_df = pd.DataFrame(
        {'class': true_labels, 'subclass': subclass_column, 'normalized_subclass': normalized_subclass_column,
         'predicted_labels': pred_labels})

    # Count how many points from each class are inside each cluster
    cluster_counts = results_df.groupby('predicted_labels')['class'].value_counts().unstack(fill_value=0)

    # Determine the dominant class for each cluster
    cluster_dominant_class = cluster_counts.idxmax(axis=1)

    # Create a mapping from predicted labels to dominant class labels
    label_mapping = {cluster: dominant_class for cluster, dominant_class in cluster_dominant_class.items()}

    # Map predicted labels to dominant class labels
    results_df['predicted_labels'] = results_df['predicted_labels'].map(label_mapping)

    # Update class names
    class_mapping = {1: 'standing', 2: 'sitting', 3: 'walking'}
    results_df['predicted_labels'] = results_df['predicted_labels'].map(class_mapping)

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

    # Plot the confusion matrix with custom colors
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(subclass_cluster_matrix, annot=True, fmt='g', cbar=False,
                     yticklabels=subclass_order, linewidths=.5, linecolor='black', cmap='Greys',
                     mask=subclass_cluster_matrix.isna(),
                     annot_kws={"weight": "bold"})  # Make numbers bold
    title = f"Subject {subject_id}"
    plt.title(title)
    plt.xlabel('Cluster Labels')
    plt.ylabel('Subclass Labels')

    # Color the background of each cell
    for i in range(subclass_cluster_matrix.shape[0]):
        for j in range(subclass_cluster_matrix.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix.iloc[i, j], edgecolor='black'))

    # Add horizontal and vertical lines to separate rows and columns
    ax.hlines(np.arange(1, len(subclass_order)), *ax.get_xlim(), color='black', linewidth=1)
    ax.vlines(np.arange(1, subclass_cluster_matrix.shape[1]), *ax.get_ylim(), color='black', linewidth=1)

    # Save the plot with high resolution if save_path is provided
    if save_path:
        save_path = os.path.join(save_path, subject_id + '.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return ari, nmi
