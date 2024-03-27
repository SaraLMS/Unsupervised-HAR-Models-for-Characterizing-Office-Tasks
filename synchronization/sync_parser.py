# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

from typing import Tuple

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
    pass

# check if logger exist and if not empty