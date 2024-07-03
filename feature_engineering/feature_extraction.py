# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import load
import parser
import tsfel
from typing import Dict, Any, List
from constants import SUPPORTED_ACTIVITIES, CABINETS, SITTING, STANDING, WALKING, STAIRS, SEC, WEAR_PREFIX

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

def generate_cfg_file(path: str) -> None:
    """
    Generates the json file from TSFEL.

    :param path: str
        Path to save the json file

    :return: None
    """
    # generate dictionary with all the features
    cfg = tsfel.get_features_by_domain()
    # path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
    # specify the full path to save the file
    file_path = os.path.join(path, "cfg_file.json")
    # save to json file
    with open(file_path, "w") as fp:
        json.dump(cfg, fp, indent=4)


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
                      json_path: str = "C:/Users/srale/PycharmProjects/toolbox/feature_engineering",
                      output_filename: str = "acc_gyr_mag_phone_features_P013.csv",
                      output_folder_name: str = "phone_features_basic_plus_activities",
                      watch_only: bool = False) -> None:
    """
    Extracts features from sensor data files contained within subfolders of a main directory, adds class and subclass
    columns based on the filenames, and saves the extracted features into a CSV file. This function also balances the
    dataset so that there's the same number of samples from each class, and for each class there's the same number of
    samples from each subclass. Each subclass should be equally sampled inside their respective class for this function
    to work correctly.

    :param data_main_path: str
        Path to the main folder. Signals are contained in the sub folders inside the main path.

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
        Name of the file containing the features

    :param output_folder_name: str
        Name of the folder in which to store the dataset

    :param watch_only: bool. Default = False:
        Extract only features from the smartwatch, removing smartphone columns.

    :return: None

    """
    # check directory
    parser.check_in_path(data_main_path, '.csv')

    # load dictionary with chosen features
    features_dict = load_json_file(json_path)

    # list to hold the dataframes
    df_dict = {}

    for folder_name in os.listdir(data_main_path):

        folder_path = os.path.join(data_main_path, folder_name)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            df = load.load_data_from_csv(file_path)

            if watch_only:
                # Filter columns that contain '_wear'
                wear_columns = [col for col in df.columns if WEAR_PREFIX in col]

                # Create a new dataframe with only the '_wear' columns
                df = df[wear_columns]

            print(f"Extract features from {folder_name}")

            # extract the features
            df = _extract_features_from_signal(df, features_dict)
            # TODO NORMALIZE FEATURES HERE

            # add class and subclass columns
            df = _add_class_and_subclass_column(df, folder_name, filename)

            # get subclass name
            subclass_name = _check_subclass(filename)

            # save in dict
            df_dict[subclass_name] = df

    # remove unwanted subclasses
    df_dict = _remove_keys_from_dict(df_dict, subclasses)

    # here guarantee that there's the same number of samples for each subclass
    all_data_df = _balance_dataset(df_dict)

    # count the number of windows of each class and subclass
    # inform user
    print(all_data_df['class'].value_counts())
    print(all_data_df['subclass'].value_counts())

    output_path = parser.create_dir(output_path, output_folder_name)
    file_path = os.path.join(output_path, output_filename)

    # save data to csv file
    all_data_df.to_csv(file_path)

    # inform user
    print(f"Data saved to {file_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #


def _remove_keys_from_dict(df_dict: Dict[str, pd.DataFrame], subclasses: List[str]) -> Dict[str, pd.DataFrame]:
    # Collect keys to be removed
    keys_to_remove = [key for key in df_dict.keys() if not any(key.startswith(subclass) for subclass in subclasses)]
    print(f"Subclasses removed: {keys_to_remove}")

    # Drop the keys/subclasses that weren't chosen
    for key in keys_to_remove:
        df_dict.pop(key)

    return df_dict


def _extract_features_from_signal(df: pd.DataFrame, features_dict: Dict[Any, Any]) -> pd.DataFrame:
    # drop time column
    if SEC in df.columns:
        df.drop(columns=[SEC], inplace=True)

    # extract the features
    features_df = tsfel.time_series_features_extractor(features_dict, df, fs=100, window_size=150, overlap=0.5)

    return features_df


# def _calculate_total_acceleration(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
#     return np.sqrt((df[columns[0]] ** 2) + (df[columns[1]] ** 2) + (df[columns[2]] ** 2))


def _add_class_and_subclass_column(df: pd.DataFrame, folder_name: str, filename: str) -> pd.DataFrame:
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


def _balance_subclasses(signals, subclass_size):
    balanced_class = [df.iloc[:subclass_size] for df in signals]
    return balanced_class


def _calculate_class_lengths(signals_classes):
    class_lengths = []
    for signals in signals_classes:
        class_length = sum(len(df) for df in signals)
        class_lengths.append(class_length)
    return class_lengths


def _balance_dataset(df_dict):
    # TODO - PUT THIS IN A FUCNTION MAYBE ?
    # TODO - SUBCLASS SIZES WHEN THERE'S STAIRS MIGHT NEED TO BE ADJUSTED BY THE USER
    # lists to store dataframes from the same class
    signals_class_1 = []
    signals_class_2 = []
    signals_class_3 = []

    # get list of signals (dataframes) from each class
    for subclass_key, df in df_dict.items():

        # df containing data from one subclass only
        # check first value of the class column since all values are the same
        if df['class'].iloc[0] == 1:
            signals_class_1.append(df)

        elif df['class'].iloc[0] == 2:
            signals_class_2.append(df)

        elif df['class'].iloc[0] == 3:
            signals_class_3.append(df)

        else:
            raise ValueError(f"Class number not supported:", df['class'].iloc[0])

    # list containing the lists of dataframes from each class of movement
    signals_classes = [signals_class_1, signals_class_2, signals_class_3]

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
            subclass_size = min_class_size // (len(signals) - 4)
        #
        # # standing class
        # elif i == 0:
        #     subclass_size = min_class_size // (len(signals))

        else:
            subclass_size = min_class_size // len(signals)

        print(f"Subclass size for class {i + 1}: {subclass_size}")
        balanced_class = _balance_subclasses(signals, subclass_size)
        all_data_list.extend(balanced_class)

    # Concatenate all balanced dataframes
    all_data_df = pd.concat(all_data_list, ignore_index=True)

    return all_data_df
