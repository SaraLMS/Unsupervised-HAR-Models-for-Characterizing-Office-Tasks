# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Tuple
import pandas as pd


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_train_test_subjects(main_path: str, subfolder_name: str, train_ration: float = 0.7) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:

    all_subjects_df = _load_subject_data(main_path, subfolder_name)
    train_subjects_df, test_subjects_df = _split_train_test_subjects(all_subjects_df, train_ration)

    return train_subjects_df, test_subjects_df


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _split_train_test_subjects(data: pd.DataFrame, train_ratio:float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets based on subjects.

    Parameters:
    - data (pd.DataFrame): DataFrame containing all subjects' data with 'subject' column.
    - train_ratio (float): Ratio of subjects to include in the training set (default is 0.7).

    Returns:
    - tuple: Two dataframes (train_data, test_data) containing train and test sets respectively.
    """
    unique_subjects = data['subject'].unique()
    num_subjects = len(unique_subjects)
    num_train_subjects = int(train_ratio * num_subjects)
    num_test_subjects = num_subjects - num_train_subjects

    train_subjects = unique_subjects[:num_train_subjects]
    test_subjects = unique_subjects[num_train_subjects:]

    train_data = data[data['subject'].isin(train_subjects)]
    test_data = data[data['subject'].isin(test_subjects)]

    return train_data, test_data


def _load_subject_data(main_path: str, subfolder_name: str) -> pd.DataFrame:
    """
    Load subject data from CSV files into a single dataframe.

    Parameters:
    - main_path (str): Root directory where subject data is stored.
    - subfolder_name (str): Name of the subfolder containing CSV files.

    Returns:
    - pd.DataFrame: Combined dataframe containing all subjects' data with 'subject' column.
    """
    dfs = []

    for subject_folder in os.listdir(main_path):
        subject_path = os.path.join(main_path, subject_folder)
        subfolder_path = os.path.join(subject_path, subfolder_name)

        if not os.path.exists(subfolder_path):
            raise FileNotFoundError(f"Subfolder '{subfolder_name}' not found in '{subject_folder}'")

        for csv_file in os.listdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, csv_file)

            df = pd.read_csv(csv_path)
            df['subject'] = subject_folder
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


