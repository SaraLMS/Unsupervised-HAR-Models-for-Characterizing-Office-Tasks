# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import load
import metrics
from constants import CLASS, SUBCLASS
from .common import normalize_features, cluster_data, check_features
from typing import List, Tuple
import os
import pandas as pd
import joblib



# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def general_model_clustering(main_path: str, subfolder_name: str, clustering_model: str,
                             feature_set: List[str], results_path: str) -> Tuple[float, float]:
    # load all subjects into a dataframe
    all_train_set, all_test_set = load.load_all_subjects(main_path, subfolder_name)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(all_train_set, feature_set)
    check_features(all_test_set, feature_set)

    # Get true labels for evaluation
    true_labels = all_test_set[CLASS]

    # Save the subclass
    subclass_column = all_test_set[SUBCLASS]

    # Normalize the features
    all_train_set = normalize_features(all_train_set)
    all_test_set = normalize_features(all_test_set)

    # Get only the wanted features in the train and test sets for the clustering
    all_train_set = all_train_set[feature_set]
    all_test_set = all_test_set[feature_set]

    # Cluster data
    pred_labels = cluster_data(clustering_model, all_train_set, all_test_set, n_clusters=3)

    # Evaluate clustering
    ri, ari, nmi = metrics.evaluate_clustering(true_labels, pred_labels)

    print(ari, nmi)

    return ari, nmi


def one_stage_general_model_each_subject(main_path: str, features_folder_name: str, clustering_model: str,
                             feature_set: List[str], results_path: str):
    # load all subjects into a dataframe
    all_train_set, _ = load.load_all_subjects(main_path, features_folder_name)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(all_train_set, feature_set)

    train_set_true_labels = all_train_set[CLASS] # y_train

    # Get only the wanted features in the train and test sets
    all_train_set = all_train_set[feature_set]

    # Normalize the features - x_train
    all_train_set = normalize_features(all_train_set)

    # Initialize the random forest classifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=2)

    # Fit the model
    rf.fit(all_train_set, train_set_true_labels)

    # lists for holding the clustering results
    results = []

    # iterate through the subject folders
    for subject_folder in os.listdir(main_path):
        subject_folder_path = os.path.join(main_path, subject_folder)

        # iterate through the folders inside each subject folder
        for folder_name in os.listdir(subject_folder_path):

            # get the specified folder
            if folder_name == features_folder_name:

                # get the path to the dataset
                features_folder_path = os.path.join(subject_folder_path, features_folder_name)

                # check if there's only one csv file in the folder
                if len(os.listdir(features_folder_path)) == 1:

                    # only one csv file for the features folder
                    dataset_path = os.path.join(features_folder_path, os.listdir(features_folder_path)[0])

                    # Train-test split
                    _, test_set = load.train_test_split(dataset_path, 0.8, 0.2)

                    check_features(test_set, feature_set)

                    # Get true labels for evaluation - y_test
                    test_set_true_labels = test_set[CLASS]

                    # get only the wanted features - x_test
                    test_set = test_set[feature_set]

                    # Normalize the features
                    test_set = normalize_features(test_set)

                    # Cluster data
                    pred_labels = cluster_data(clustering_model, all_train_set, test_set, n_clusters=3)

                    # Predict on the test set
                    y_pred = rf.predict(test_set)

                    # Calculate the accuracy
                    accuracy = accuracy_score(test_set_true_labels, y_pred)

                    # Evaluate clustering
                    ri, ari, nmi = metrics.evaluate_clustering(test_set_true_labels, pred_labels)

                    results.append({
                        "Subject ID": subject_folder,
                        "ARI": ari,
                        "NMI": nmi,
                        "Random Forest": accuracy
                    })

    # Create DataFrame from results and save to Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(results_path, "1_stage_general_kmeans_basic_watch_phone.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # save_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels/all_activities_1_stage_general_model_RF.joblib"
    # joblib.dump(rf, save_path)
    # print(f"Model saved to {save_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #



