# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from typing import Tuple
import pandas as pd
from .load_sync_data import load_data_from_csv


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #
def load_all_subjects(main_path: str, subfolder_name: str) -> pd.DataFrame:
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

            df = load_data_from_csv(csv_path)
            df['subject'] = subject_folder
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #



