# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from pathlib import Path
import re

from constants import SUBJECT_PREFIX


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_folder_name_from_path(folder_path: str) -> str:
    """
    gets the folder name from the folder path.

    Parameters:

    folder_path (str):
    Path to the folder containing the data.

    Returns:
        Folder name in the folder path
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    return folder_name


def get_subject_id_from_path(file_path: str) -> str:
    """
    Gets the subject id from a path.
    For this function to work, there should be at least one folder in the path in which the folder name is the subject id.

    :param file_path: str
    Path to the csv file

    :return: str
    String that corresponds to subject id, found in the path
    """
    # search for the pattern of the subject id
    pattern = rf'{re.escape(SUBJECT_PREFIX)}\d+'

    # Search for the pattern in the file path
    match = re.search(pattern, file_path)

    if match:
        # If a match is found, return the matched string
        return match.group(0)
    else:
        # If no match is found, return an empty string or raise an error
        raise ValueError(f"Subject ID not found in the path: {file_path}")
