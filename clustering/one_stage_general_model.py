# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import load
import metrics
from constants import CLASS, SUBCLASS
from .common import normalize_features, cluster_data, check_features
from typing import List, Tuple
import os
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def general_model_clustering(main_path: str, subfolder_name: str, clustering_model: str,
                             feature_set: List[str], results_path: str) -> Tuple[float, float]:
    # load all subjects into a dataframe
    train_set, test_set = load.load_all_subjects(main_path, subfolder_name, False)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(train_set, feature_set)
    check_features(test_set, feature_set)

    # Get true labels for evaluation
    true_labels = test_set[CLASS]

    # Save the subclass
    subclass_column = test_set[SUBCLASS]

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

    print(ari, nmi)

    return ari, nmi

    # # Normalize features (excluding subject, class, and subclass)
    # feature_columns = feature_set.copy()
    # all_subjects_df[feature_columns] = normalize_features(all_subjects_df[feature_columns])
    #
    # # Initialize evaluation metrics accumulators
    # total_adj_rand_index = 0.0
    # total_norm_mutual_info = 0.0
    #
    # results = []
    # num_subjects = len(all_subjects_df['subject'].unique())
    #
    # # Perform leave-one-out cross-validation
    # for i, test_subject in enumerate(all_subjects_df['subject'].unique(), 1):
    #     print(f"Testing with subject {test_subject} (Iteration {i}/{num_subjects})")
    #
    #     # Split into train and test data
    #     train_subjects_df = all_subjects_df[all_subjects_df['subject'] != test_subject]
    #     test_subjects_df = all_subjects_df[all_subjects_df['subject'] == test_subject]
    #
    #     # Keep a copy of true labels before dropping them from the dataframe
    #     true_labels = test_subjects_df['class'].copy()
    #
    #     # Drop class, subclass, and subject columns for clustering
    #     train_subjects_df = train_subjects_df.drop(['class', 'subclass', 'subject'], axis=1)
    #     test_subjects_df = test_subjects_df.drop(['class', 'subclass', 'subject'], axis=1)
    #
    #     # Perform clustering based on the selected model
    #     labels = cluster_data(clustering_model, train_subjects_df, test_subjects_df, n_clusters=3)
    #
    #     # Evaluate clustering
    #     rand_index, adj_rand_index, norm_mutual_info = metrics.evaluate_clustering(true_labels, labels)
    #
    #     results.append({
    #         "Subject ID": test_subject,
    #         "ARI": adj_rand_index,
    #         "NMI": norm_mutual_info
    #     })
    #
    #     print(f"results for {test_subject}")
    #     print(f"ari: {adj_rand_index}, nmi: {norm_mutual_info}")
    #
    #     total_adj_rand_index += adj_rand_index
    #     total_norm_mutual_info += norm_mutual_info
    #
    # # Create DataFrame from results and save to Excel
    # results_df = pd.DataFrame(results)
    # excel_path = os.path.join(results_path, "leave_one_out_basic_watch_phone_kmeans.xlsx")
    # results_df.to_excel(excel_path, index=False)
    #
    # avg_adj_rand_index = total_adj_rand_index / num_subjects
    # avg_norm_mutual_info = total_norm_mutual_info / num_subjects
    #
    # print(f"Average clustering results over {num_subjects} subjects:\n"
    #       f"Avg Adjusted Rand Index: {avg_adj_rand_index}\n"
    #       f"Avg Normalized Mutual Information: {avg_norm_mutual_info}")
    #
    # return avg_adj_rand_index, avg_norm_mutual_info

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #



