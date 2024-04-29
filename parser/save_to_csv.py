# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import os
import pandas as pd
from parser.check_create_directories import create_dir

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def save_data_to_csv(output_filename: str, signals_df: pd.DataFrame, output_path: str, folder_name: str) -> None:
    """
    Creates directory and saves sensor data to a csv file.

    Parameters:
        output_filename (str):
        The name for the output file.

        signals_df (pd.DataFrame):
        DataFrame containing the synchronized sensor data to be saved to a csv file

        output_path (str):
        Location where the file should be saved.

        folder_name (str):
        The name of the folder containing the data.
    """
    # get folder name without _
    folder = folder_name.split('_')

    # create dir
    output_path = create_dir(output_path, folder[0])

    # add filename to get full path
    output_path = os.path.join(output_path, output_filename)

    # save dataframe to csv file
    signals_df.to_csv(output_path)