# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
import os
import pandas as pd
import joblib

# internal imports
import load
import metrics
from constants import CLASS, SUBCLASS, AGGLOMERATIVE, SUPPORTED_MODELS
from .common import cluster_data, check_features


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def one_stage_general_model_each_subject(main_path: str, features_folder_name: str, clustering_model: str, nr_clusters: int,
                                         feature_set: List[str], results_path: str) -> None:
    """
    This function implements experiment 2 for the one-stage general model. For more information check README file.

    In this experiment, clustering is performed on the train set (of all subjects combined) and the test set instances
    are appointed to the preformed clusters (each subject individually).
    If Agglomerative Clustering is chosen, clustering is performed on 20 % of the data.
    This function also trains a random forest with the same feature sets used for the clustering.

    This saves the adjusted rand index, normalized mutual information, and accuracy score (random forest) results in
    an Excel sheet.
    :param main_path: str
    Path to the main_folder containing subfolders which have the datasets. The directory scheme is the following
    main_folder/folder/subfolder/dataset.csv
    (i.e., datasets/features_basic_acc_gyr_mag_phone_watch/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)

    :param features_folder_name: str
    Path to the folder[*] identifying which datasets to load. The directory scheme is the following
    folder[*]/subfolder/dataset.csv
    (i.e., features_basic_acc_gyr_mag_phone_watch[*]/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :param nr_clusters: int
    Number of clusters to find

    :param feature_set: List[str]
    List containing the feature names to be used for clustering

    :param results_path: str
     Path to the directory where to save the Excel sheet with the clustering and random forest results.

    :return: None
    """
    # check supported models
    _check_supported_models(clustering_model)
    # load all subjects into a dataframe
    all_train_set, _ = load.load_all_subjects(main_path, features_folder_name)

    # Check if all features in the feature set exist in the dataframe columns
    check_features(all_train_set, feature_set)

    train_set_true_labels = all_train_set[CLASS]  # y_train

    # Get only the wanted features in the train and test sets
    all_train_set = all_train_set[feature_set]

    # Normalize the features - x_train
    # Initialize scaler
    scaler = MinMaxScaler()

    # scale the train set
    all_train_set = scaler.fit_transform(all_train_set)

    # put the train and test sets back into a pandas dataframe
    all_train_set = pd.DataFrame(all_train_set, columns=feature_set)

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

                    # scale the test set with the same normalization parameters as the train set
                    test_set = scaler.transform(test_set)
                    test_set = pd.DataFrame(test_set, columns=feature_set)

                    # Cluster data
                    if clustering_model == AGGLOMERATIVE:
                        pred_labels = cluster_data(clustering_model, all_train_set, nr_clusters)

                    else:
                        pred_labels = cluster_data(clustering_model, all_train_set, nr_clusters, test_set)

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
    excel_path = os.path.join(results_path, "1_stage_general_agg_all_phone.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")

    # save_path = "C:/Users/srale/OneDrive - FCT NOVA/Tese/excels/all_activities_1_stage_general_model_RF.joblib"
    # joblib.dump(rf, save_path)
    # print(f"Model saved to {save_path}")

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _check_supported_models(clustering_model: str):
    """
    Check is the chosen model is in the supported sensors. If not, raises a value error.

    :param clustering_model: str
    Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model

    :return: None
    """
    if clustering_model not in SUPPORTED_MODELS:
        raise ValueError(f"{clustering_model} is no supported. Supported models are: {SUPPORTED_MODELS}")