# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from typing import Tuple
import glob
import os

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #


def extract_date_time(file_path: str) -> Tuple[str, str]:
    """
    extracts the date and the time from the file path
    :param file_path: the path of the file
    :return: the time and the date
    """
    # get the date and the time from the path string
    date_time = file_path.rsplit('.', 1)[0].rsplit('_', 2)

    # extract date and time
    date = date_time[1]
    time = date_time[2]

    time = time.split('-')
    time = ':'.join(time)

    return date, time


def check_logger_file(folder_path:str) -> bool:
    """
    Checks if a logger file exists in the specified folder and that it is not empty.
    Assumes logger file name starts with 'opensignals_ACQUISITION_LOG_' and includes
    a timestamp.

    Parameters:
    ----------
    folder_path : str
        The path to the folder containing the RAW acquisitions.

    Returns:
    -------
    True if the logger file exists and is not empty, False otherwise.


    """

    # Pattern to match the logger file, assuming it starts with 'opensignals_ACQUISITION_LOG_'
    pattern = os.path.join(folder_path, 'opensignals_ACQUISITION_LOG_*')

    # Use glob to find files that match the pattern
    matching_files = glob.glob(pattern)

    # Check if there's at least one matching file and that the first matching file is not empty
    if matching_files and os.path.getsize(matching_files[0]) > 0:
        return True
    else:
        return False
