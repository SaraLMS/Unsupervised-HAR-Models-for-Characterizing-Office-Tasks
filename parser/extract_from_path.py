# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from pathlib import Path


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def get_folder_name_from_path(folder_path: str) -> str:
    """
    gets folder name from folder path.

    Parameters:

    folder_path (str):
    Path to the folder containing the data.

    Returns:
        Folder name in the folder path
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    return folder_name
