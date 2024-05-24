# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import os

from constants import STANDING, CABINETS, SITTING, WALKING, SUPPORTED_ACTIVITIES
COFFEE = "coffee"
FOLDERS = "folders"
SIT = "sit"
GESTURES = "gestures"
NO_GESTURES = "no_gestures"
FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
SUPPORTED_SUBCLASSES = [COFFEE, FOLDERS, SIT, NO_GESTURES, GESTURES, FAST, MEDIUM, SLOW]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_part_from_csv(file_path: str, portion: int) -> pd.DataFrame:
    if not 0 < portion <= 100:
        raise ValueError("Portion must be between 1 and 100.")

    # Read the CSV file
    df = pd.read_csv(file_path, index_col=0)

    # Calculate the index for the specified portion of the data
    portion_index = int(len(df) * (portion / 100))

    # Extract the portion of the data
    df = df.iloc[:portion_index]

    return df


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #



