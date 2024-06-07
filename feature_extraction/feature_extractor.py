# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from load.load_sync_data import load_data_from_csv
from parser.save_to_csv import save_data_to_csv
from tsfel.feature_extraction.features_settings import get_features_by_domain
from tsfel.feature_extraction.calc_features import time_series_features_extractor
from constants import SUPPORTED_ACTIVITIES, CABINETS, SITTING, STANDING, WALKING, STAIRS
from parser.check_create_directories import check_in_path

# constants supported from filenames
COFFEE = "coffee"
FOLDERS = "folders"
SIT = "sit"
GESTURES = "gestures"
STAND_STILL = "stand_still"
FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
STAIRS_UP = "stairsup"
STAIRS_DOWN = "stairsdown"
SUPPORTED_SUBCLASSES = [COFFEE, FOLDERS, SIT, STAND_STILL, GESTURES, FAST, MEDIUM, SLOW, STAIRS_UP, STAIRS_DOWN]


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def generate_cfg_file(path: str):
    # generate dictionary with all the features
    cfg = get_features_by_domain()
    # path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
    # specify the full path to save the file
    file_path = os.path.join(path, "cfg_file.json")
    # save to json file
    with open(file_path, "w") as fp:
        json.dump(cfg, fp, indent=4)


def feature_extractor(data_main_path: str, output_path: str, subclasses: list[str],
                      json_path: str = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction",
                      output_filename: str = "mag_phone_watch_plus_spectralP005.csv", output_folder_name: str = "features",
                      total_acceleration: bool = False) -> None:

    # TODO - DOCSTRING THIS SHIT
    # check directory
    check_in_path(data_main_path, '.csv')

    json_file_path = os.path.join(json_path, "cfg_file.json")
    # read json file to a features dict
    with open(json_file_path, "r") as file:
        features_dict = json.load(file)

    # list to hold the dataframes
    df_dict = {}

    for folder_name in os.listdir(data_main_path):

        folder_path = os.path.join(data_main_path, folder_name)

        for filename in os.listdir(folder_path):

            file_path = os.path.join(folder_path, filename)

            df = load_data_from_csv(file_path)

            # if total_acceleration:
            #     # check if there's phone accelerometer or watch accelerometer
            #     phone_acc_columns = []
            #     watch_acc_columns = []
            #
            #     # Separate the columns by device
            #     for col in df.columns:
            #         if ACCELEROMETER_PREFIX in col:
            #             if WEAR_PREFIX in col:
            #                 watch_acc_columns.append(col)
            #             else:
            #                 phone_acc_columns.append(col)
            #
            #     # Check if accelerometer data is available for phone and calculate total acceleration
            #     if phone_acc_columns:
            #         df['total_acc_phone'] = _calculate_total_acceleration(df, phone_acc_columns)
            #
            #     # Check if accelerometer data is available for smartwatch and calculate total acceleration
            #     if watch_acc_columns:
            #         df['total_acc_wear'] = _calculate_total_acceleration(df, watch_acc_columns)

            print(f"Extract features from {folder_name}")

            # extract the features
            df = _extract_features_from_signal(df, features_dict)

            # add class and subclass columns
            df = _add_class_and_subclass_column(df, folder_name, filename)

            # get subclass name
            subclass_name = _check_subclass(filename)

            # save in dict
            df_dict[subclass_name] = df

    # Collect keys to be removed
    keys_to_remove = [key for key in df_dict.keys() if key not in subclasses]

    # drop the signals/ subclasses that weren't chosen
    for key in keys_to_remove:
        df_dict.pop(key)

    # here guarantee that there's the same number of samples for each subclass
    all_data_df = _balance_dataset(df_dict)

    # count the number of windows of each class and subclass
    # inform user
    print(all_data_df['class'].value_counts())
    print(all_data_df['subclass'].value_counts())

    # save data to csv file
    save_data_to_csv(output_filename, all_data_df, output_path, output_folder_name)

    # inform user
    print(f"Data saved to {output_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _extract_features_from_signal(df: pd.DataFrame, features_dict: Dict[Any, Any]) -> pd.DataFrame:
    # drop time column
    df.drop(columns=['sec'], inplace=True)

    # extract the features
    features_df = time_series_features_extractor(features_dict, df, fs=100, window_size=150, overlap=0.5)

    return features_df


def _calculate_total_acceleration(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return np.sqrt((df[columns[0]] ** 2) + (df[columns[1]] ** 2) + (df[columns[2]] ** 2))


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
            f"The activity: {folder_name} is not supported. Supported activities are: {SUPPORTED_ACTIVITIES}")

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

    elif STAND_STILL in filename:
        subclass_str = "standing_still"

    elif SLOW in filename:
        subclass_str = "walk_slow"

    elif MEDIUM in filename:
        subclass_str = "walk_medium"

    elif FAST in filename:
        subclass_str = "walk_fast"

    elif STAIRS_UP in filename:
        subclass_str = "stairs_up"

    elif STAIRS_DOWN in filename:
        subclass_str = "stairs_down"

    else:
        raise ValueError(f"Subclass not supported. Supported subclasses are: ")

    return subclass_str


def _calculate_class_lengths(signals_classes):
    class_lengths = []
    for signals in signals_classes:
        class_length = sum(len(df) for df in signals)
        class_lengths.append(class_length)
    return class_lengths


def _balance_subclasses(signals, subclass_size):
    balanced_class = [df.iloc[:subclass_size] for df in signals]
    return balanced_class


def _balance_dataset(df_dict):
    # TODO - PUT THIS IN A FUCNTION MAYBE ?
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
            subclass_size = (min_class_size // (len(signals) - 1)) - 20
        else:
            subclass_size = min_class_size // len(signals)

        print(f"Subclass size for class {i + 1}: {subclass_size}")
        balanced_class = _balance_subclasses(signals, subclass_size)
        all_data_list.extend(balanced_class)

    # Concatenate all balanced dataframes
    all_data_df = pd.concat(all_data_list, ignore_index=True)

    return all_data_df
