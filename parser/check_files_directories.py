# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import os
from glob import glob


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def check_in_path(raw_data_in_path: str) -> None:
    """
    Checks if the specified path is valid according to the criteria:
    - The path exists and is a directory.
    - Contains subdirectories.
    - Each subdirectory contains at least one .txt file.

    Parameters:
    raw_data_in_path (str):
    The main folder path containing subfolders with raw sensor data.

    Raises:
    ValueError: If any of the criteria are not met.
    """
    if not os.path.isdir(raw_data_in_path):
        raise ValueError(f"The path {raw_data_in_path} does not exist or is not a directory.")

    subfolders = [f.path for f in os.scandir(raw_data_in_path) if f.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in the main path {raw_data_in_path}.")

    for subfolder in subfolders:
        txt_files = glob(os.path.join(subfolder, "*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in subfolder {subfolder}.")
