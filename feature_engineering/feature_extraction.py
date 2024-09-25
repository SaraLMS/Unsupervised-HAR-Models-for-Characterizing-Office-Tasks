# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
import os
import numpy as np
import pandas as pd
import tsfel
from typing import Dict, Any, List

# internal imports
import load
import parser
from constants import SUPPORTED_ACTIVITIES, CABINETS, SITTING, STANDING, WALKING, STAIRS, SEC, WEAR_PREFIX, \
    MAGNETOMETER_PREFIX, CSV

# constants supported from filenames
COFFEE = "coffee"
FOLDERS = "folders"
SIT = "sit"
GESTURES = "gestures"
STANDING_GESTURES = "standing_gestures"
STAND_STILL1 = "stand_still1"
STAND_STILL2 = "stand_still2"
STANDING_STILL = "standing_still"
STAIRS = "stairs"
WALK_FAST = "walk_fast"
WALK_MEDIUM = "walk_medium"
WALK_SLOW = "walk_slow"
FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
STAIRS_UP1 = "stairsup1"
STAIRS_DOWN1 = "stairsdown1"
STAIRS_UP2 = "stairsup2"
STAIRS_DOWN2 = "stairsdown2"
STAIRS_UP3 = "stairsup3"
STAIRS_DOWN3 = "stairsdown3"
STAIRS_UP4 = "stairsup4"
STAIRS_DOWN4 = "stairsdown4"

SUPPORTED_FILENAME_SUBCLASSES = [COFFEE, FOLDERS, SIT, STAND_STILL1, GESTURES, FAST,
                                 MEDIUM, SLOW, STAIRS_UP1, STAIRS_DOWN1, STAIRS_UP2, STAIRS_DOWN2, STAIRS_UP3,
                                 STAIRS_DOWN3, STAIRS_UP4, STAIRS_DOWN4]

SUPPORTED_INPUT_SUBCLASSES = [SIT, COFFEE, FOLDERS, STANDING_STILL, STANDING_GESTURES, WALK_FAST, WALK_SLOW,
                              WALK_MEDIUM, STAIRS]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def load_json_file(json_path: str) -> Dict[Any, Any]:
    """
    Loads the json file containing the features from TSFEL to a dictionary
    :param json_path: str
        Path to the json file
    :return: Dict[Any,Any]
    Dictionary containing the features from TSFEL
    """
    json_file_path = os.path.join(json_path, "cfg_file.json")
    # read json file to a features dict
    with open(json_file_path, "r") as file:
        features_dict = json.load(file)

    return features_dict


def feature_extractor(data_main_path: str, output_path: str, subclasses: list[str],
                      output_filename: str, output_folder_name: str,
                      json_path: str = "C:/Users/srale/PycharmProjects/toolbox/feature_engineering") -> None:
    """
    # TODO filename prefix and subject number - FILE_SUFFIX = "_feature_P{}.csv" in the constants
    file_name = prefix + FILE_SUFFIX.format(subject_num)
    # TODO doctsring sucks

    Extracts features from sensor data files contained within the sub-folders of a main directory, adds class and subclass
    columns based on the filenames, and saves the extracted features into a CSV file. This function also balances the
    dataset so that there's the same number of samples from each class, and for each class there's the same number of
    samples from each subclass. Each subclass should be equally sampled inside their respective class for this function
    to work correctly.

    :param data_main_path: str
        Path to the main folder. Signals are contained in the sub folders inside the main path. EXAMPLE

    :param output_path: str
        Path to the folder where the csv file should be saved.

    :param subclasses: List[str]
        List containing the name of the subclasses to load and extract features. Supported subclasses:
            "sit": sitting
            "standing_still": Standing still
            "standing_gestures": Standing with gestures
            "coffee": Standing while doing coffee
            "folders": Standing while moving folders inside a cabinet
            "walk_slow": Walking slow speed
            "walk_medium": Walking medium speed
            "walk_fast": Walking fast speed
            "stairs": Going up and down the stairs

    :param json_path: str
        Path to the json file containing the features to be extracted using TSFEL

    :param output_filename: str
        Name of the file containing the EXTRACTED features

    :param output_folder_name: str
        Name of the folder in which to store the dataset

    :return: None

    """
    # check if subclasses is valid
    _validate_subclasses_list(subclasses)

    # check directory
    parser.check_in_path(data_main_path, CSV)

    # load dictionary with chosen features
    features_dict = load_json_file(json_path)

    # list to hold the dataframes
    df_dict = {}

    for folder_name in os.listdir(data_main_path):

        folder_path = os.path.join(data_main_path, folder_name)

        for filename in os.listdir(folder_path):

            # get subclass name from the filename
            subclass_name = _check_subclass(filename)

            # load only the chosen subclasses
            if any(subclass in subclass_name for subclass in subclasses):
                # get the path to the csv with the signals from that subclass
                file_path = os.path.join(folder_path, filename)

                # load to a dataframe
                df = load.load_data_from_csv(file_path)

                # get magnitude of the magnetometer
                df = _calc_mag_magnitude(df)

                # inform user
                print(f"Extract features from {folder_name}")

                # extract the features
                df = _extract_features_from_signal(df, features_dict)

                # add class and subclass columns
                df = _add_class_and_subclass_column(df, folder_name, filename)

                # save in dict
                df_dict[subclass_name] = df

    # here guarantee that there's the same number of samples for each subclass
    all_data_df = _balance_dataset(df_dict)

    # count the number of windows of each class and subclass
    # inform user
    print(all_data_df['class'].value_counts())
    print(all_data_df['subclass'].value_counts())

    output_path = parser.create_dir(output_path, output_folder_name)
    file_path = os.path.join(output_path, output_filename)
    #
    # save data to csv file
    all_data_df.to_csv(file_path)

    # inform user
    print(f"Data saved to {file_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _calc_mag_magnitude(df:pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the magnitude of the magnetometer for both the smartphone and smartwatch

    :param df: pd.DataFrame
    pandas DataFrame containing the MAG signals

    :return: pd.DataFrame
    A pandas DataFrame containing the new columns with the MAG magnenitude
    """
    phone_mag_columns = []
    watch_mag_columns = []

    # Separate the columns by device
    for col in df.columns:
        if MAGNETOMETER_PREFIX in col:
            if WEAR_PREFIX in col:
                watch_mag_columns.append(col)
            else:
                phone_mag_columns.append(col)

    # Calculate magnitude if accelerometer data is available
    if phone_mag_columns:
        df['total_mag_phone'] = _calculate_magnitude(df, phone_mag_columns)

    if watch_mag_columns:
        df['total_mag_wear'] = _calculate_magnitude(df, watch_mag_columns)

    # # Optionally drop the original magnetometer columns
    # df.drop(columns=phone_mag_columns + watch_mag_columns, inplace=True)

    return df


def _calculate_magnitude(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    Calculate magnitude of sensor data

    :param df: pd.DataFrame
    pandas DataFrame containing the sensor data

    :param columns: List[str]
    list with column names to calculate the magnitude

    :return: np.ndarray
    A numpy array containing the magnitude of the chosen columns of the input dataframe
    """
    return np.sqrt((df[columns[0]] ** 2) + (df[columns[1]] ** 2) + (df[columns[2]] ** 2))


def _validate_subclasses_list(subclasses: List[str]) -> None:
    """
    Raises ValueError exceptions if the subclasses list is empty or if the chosen subclasses are not supported.

    :param subclasses: List[str]
    List containing the subclass strings

    :return: None
    """
    # Raise an exception if the subclasses list is empty
    if not subclasses:
        raise ValueError("The subclasses list is empty. Please provide at least one subclass.")

    # check if the subclasses are valid
    for subclass in subclasses:
        if subclass not in SUPPORTED_INPUT_SUBCLASSES:
            raise ValueError(f"Subclass: {subclass} is not supported.\n "
                             f"Supported subclasses are: {SUPPORTED_INPUT_SUBCLASSES}")


def _extract_features_from_signal(df: pd.DataFrame, features_dict: Dict[Any, Any]) -> pd.DataFrame:
    """
    Remove the time column ('sec') and extract features from sensor data contained in the columns of a pandas DataFrame,
     using TSFEL.

    :param df: pd.DataFrame
    pandas DataFrame containing the signals in each column

    :param features_dict: Dict[Any, Any]
    Dictionary containing the features to be extracted. For more info check TSFEL documentation
    (https://tsfel.readthedocs.io/en/latest/descriptions/get_started.html#set-up-the-feature-extraction-config-file)

    :return: pd.DataFrame
    A pandas DataFrame with the extracted features. Each row is an instance and columns are the features
    """
    # drop time column
    if SEC in df.columns:
        df.drop(columns=[SEC], inplace=True)

    # extract the features
    features_df = tsfel.time_series_features_extractor(features_dict, df, fs=100, window_size=150, overlap=0.5)

    return features_df


def _add_class_and_subclass_column(df: pd.DataFrame, folder_name: str, filename: str) -> pd.DataFrame:
    """
    Adds a class column based on the folder name where the signals are stored and a subclass column
    based on the filename.

    :param df: pd.DataFrame
    pandas DataFrame containing the features in the columns and instances in the rows

    :param folder_name: str
    Name of the folder (i.e., WALKING) from which the signals are loaded
    (i.e. main_folder/subfolder/acc_gyr_mag_phone/P001/WALKING/segment_signals.csv)

    :param filename: str
    Name of the file containing the sensor data (i.e. segment_signals.csv)
    (i.e. main_folder/subfolder/acc_gyr_mag_phone/P001/WALKING/segment_signals.csv)

    :return: pd.DataFrame
    A pandas DataFrame containing the new class and subclass columns
    """
    # attribute class and subclass numbers according to the activity and tasks
    if CABINETS in folder_name or STANDING in folder_name:
        class_number = 1
        subclass_str = _check_subclass(filename)

    elif SITTING in folder_name:
        class_number = 2
        subclass_str = _check_subclass(filename)

    elif STAIRS in folder_name or WALKING in folder_name:
        class_number = 3
        subclass_str = _check_subclass(filename)

    else:
        raise ValueError(
            f"The activity: {folder_name} is not supported. \n Supported activities are: {SUPPORTED_ACTIVITIES}")

    # add class column
    df['class'] = class_number

    # add subclass column
    df['subclass'] = subclass_str

    return df


def _check_subclass(filename: str) -> str:
    """
    Check which subclass the signal belongs to based on the suffixes on the filename

    :param filename: str
    Name of the file containing the sensor data (i.e. segment_signals.csv)
    (i.e. main_folder/subfolder/acc_gyr_mag_phone/P001/WALKING/segment_signals.csv)

    :return: str
    Subclass name based on the suffix found on the filename
    """
    # check subclass
    if COFFEE in filename:
        subclass_str = "standing_coffee"

    elif FOLDERS in filename:
        subclass_str = "standing_folders"

    elif SIT in filename:
        subclass_str = "sit"

    elif GESTURES in filename:
        subclass_str = "standing_gestures"

    elif STAND_STILL1 in filename:
        subclass_str = "standing_still1"

    elif STAND_STILL2 in filename:
        subclass_str = "standing_still2"

    elif SLOW in filename:
        subclass_str = "walk_slow"

    elif MEDIUM in filename:
        subclass_str = "walk_medium"

    elif FAST in filename:
        subclass_str = "walk_fast"

    elif STAIRS_UP1 in filename:
        subclass_str = "stairs_up1"

    elif STAIRS_DOWN1 in filename:
        subclass_str = "stairs_down1"

    elif STAIRS_UP2 in filename:
        subclass_str = "stairs_up2"

    elif STAIRS_DOWN2 in filename:
        subclass_str = "stairs_down2"

    elif STAIRS_UP3 in filename:
        subclass_str = "stairs_up3"

    elif STAIRS_UP4 in filename:
        subclass_str = "stairs_up4"

    elif STAIRS_DOWN3 in filename:
        subclass_str = "stairs_down3"

    elif STAIRS_DOWN4 in filename:
        subclass_str = "stairs_down4"

    else:
        raise ValueError(f"Subclass not supported. Check filename: {filename}"
                         f" \n Supported subclasses are: {SUPPORTED_FILENAME_SUBCLASSES}")

    return subclass_str


def _balance_subclasses(signals: List[pd.DataFrame], subclass_size: int) -> List[pd.DataFrame]:
    """
    Cuts the dataframes inside a list of dataframes to the length of subclass_size. The dataframes are cut at the start.

    :param signals: List[pd.DataFrame]
    List of dataframes

    :param subclass_size: Int
    Length in samples to cut from the start of the dataframes

    :return: List[pd.DataFrame]
    List of dataframes cut to the same length
    """
    balanced_class = [df.iloc[:subclass_size] for df in signals]
    return balanced_class


def _calculate_class_lengths(signals_classes: List[List[pd.DataFrame]]) -> List[int]:
    """
    Calculates the number of instances from each class by counting the instances of each subclass belonging to the
    class.

    :param signals_classes: List[List[pd.DataFrame]]
    List containing lists of all the signals belonging to the same class.
    (i.e., [[walk_slow_df, walk_medium_df, walk_fast_df], [stand_still_df, stand_gestures_df], [sit_df]]

    :return: List[int]
    A list of integers representing the number of instances from each class
    """
    class_lengths = []
    for signals in signals_classes:
        class_length = sum(len(df) for df in signals)
        class_lengths.append(class_length)
    return class_lengths


def _balance_dataset(df_dict: Dict[str, pd.DataFrame]):
    """
    Balances the dataset so that there's roughly the same amount of instances from the same class, and inside each
    class, there's the same amount of instances from the same subclass. After balancing, joins all the data into one
    single dataframe

    :param df_dict: Dict[str, pd.DataFrame]
    Dictionary where the keys are the subclass names and the values are the dataframes containing the features extracted
    from the respective signals.

    :return: pd.DataFrame
    A pandas DataFrame containing the balanced classes and subclasses
    """

    # lists to store dataframes from the same class
    signals_standing = []
    signals_sitting = []
    signals_walking = []
    # TODO: @p-probst dynamically assign classes
    # get list of signals (dataframes) from each class
    for subclass_key, df in df_dict.items():
        # TODO APPEND LEN OF DF
        # df containing data from one subclass only
        # check first value of the class column since all values are the same
        if df['class'].iloc[0] == 1:
            signals_standing.append(df)

        elif df['class'].iloc[0] == 2:
            signals_sitting.append(df)

        elif df['class'].iloc[0] == 3:
            signals_walking.append(df)

        else:
            raise ValueError(f"Class number not supported:", df['class'].iloc[0])

    # list containing the lists of dataframes from each class of movement
    signals_classes = [signals_standing, signals_sitting, signals_walking]

    # Calculate the lengths of each class
    class_lengths = _calculate_class_lengths(signals_classes)
    print("Class lengths:", class_lengths)

    # Determine the minimum class size
    min_class_size = min(class_lengths)
    print("Min class size:", min_class_size)

    # Balance each class
    all_data_list = []

    # Check if "stairs" is in any of the keys
    stairs_present = any("stairs" in key for key in df_dict.keys())

    for i, signals in enumerate(signals_classes):

        # Special case for class 3 if stairs are present, subclass size needs adjustments
        if stairs_present and i == 2:
            subclass_size = min_class_size // (len(signals)-2)
        #
        # # standing class
        elif i == 0:
            subclass_size = min_class_size // (len(signals))

        else:
            subclass_size = min_class_size // len(signals)

        print(f"Subclass size for class {i + 1}: {subclass_size}")
        balanced_class = _balance_subclasses(signals, subclass_size)
        all_data_list.extend(balanced_class)

    # Concatenate all balanced dataframes
    all_data_df = pd.concat(all_data_list, ignore_index=True)

    return all_data_df
