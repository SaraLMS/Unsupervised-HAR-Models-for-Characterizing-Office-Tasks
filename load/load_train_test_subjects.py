# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
import pandas as pd

from constants import SUBJECT
from .load_sync_data import load_data_from_csv
from .dataset_split_train_test import train_test_split


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_all_subjects(main_path: str, subfolder_name: str, all_data: bool, train_size: float = 0.8,
                      test_size: float = 0.2) -> pd.DataFrame:
    """
    Load subject data from CSV files into a unified dataframe. If all_data is set to True, load the complete dataset of
    each subject; otherwise, load only the training sets. A 'subject' column is added for identifying subjects.
    The training set consists of the initial portion of the entire dataset, determined by the train_size parameter.

    :param main_path: str
    Root directory where all subjects data is stored

    :param subfolder_name: str
    Name of the subfolder containing the dataset to be loaded

    :param all_data: bool
    If set to True, the function loads all the data from each subject. If False, loads only the train_set of all subjects

    :param train_size: float (Optional)
    Size of the train set.

    :param test_size: float (Optional)
    Size of the test set

    :return: pd.DataFrame
    A pandas dataframe containing the data from all subjects including a new 'subject' column for subject identification
    """
    dfs_train = []
    dfs_test = []
    dfs = []

    for subject_folder in os.listdir(main_path):
        subject_path = os.path.join(main_path, subject_folder)
        subfolder_path = os.path.join(subject_path, subfolder_name)

        if not os.path.exists(subfolder_path):
            raise FileNotFoundError(f"Subfolder '{subfolder_name}' not found in '{subject_folder}'")

        for csv_file in os.listdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, csv_file)

            if not all_data:

                # get only the train sets from each subject
                train_set, test_set = train_test_split(csv_path, train_size, test_size)

                # add subject column for subject identification
                # train_set[SUBJECT] = subject_folder
                # test_set[SUBJECT] = subject_folder

                # add to the list of dataframes
                dfs_train.append(train_set)
                dfs_test.append(test_set)

            else:

                # get the whole dataset from all subjects
                df = load_data_from_csv(csv_path)

                # add subject column for subject identification
                df[SUBJECT] = subject_folder

                # add to the list of dataframes
                dfs.append(df)

    return pd.concat(dfs_train, ignore_index=True), pd.concat(dfs_test, ignore_index=True),


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #



