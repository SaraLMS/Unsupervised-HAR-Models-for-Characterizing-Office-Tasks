# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import json
import os
from typing import Dict, Any
import pandas as pd
import tsfel
from constants import SUPPORTED_ACTIVITIES, CABINETS, SITTING, STANDING, WALKING
from load.load_sync_data import load_data_from_csv
from parser.check_create_directories import check_in_path

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

def generate_cfg_file():
    # generate dictionary with all the features
    cfg = tsfel.get_features_by_domain()
    path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
    # specify the full path to save the file
    file_path = os.path.join(path, "cfg_file.json")
    # save to json file
    with open(file_path, "w") as fp:
        json.dump(cfg, fp, indent=4)


def feature_extractor(data_main_path: str, output_path: str,
                      output_filename: str = "P002_watch_acc_phone_acc_gyr_dataset.csv") -> None:
    # check directory
    check_in_path(data_main_path, '.csv')

    path = "C:/Users/srale/PycharmProjects/toolbox/feature_extraction"
    file_path = os.path.join(path, "cfg_file.json")
    # read json file to a features dict
    with open(file_path, "r") as file:
        features_dict = json.load(file)

    # list to hold the dataframes
    df_list = []

    for folder_name in os.listdir(data_main_path):

        folder_path = os.path.join(data_main_path, folder_name)

        for filename in os.listdir(folder_path):
            # get file_path
            file_path = os.path.join(folder_path, filename)

            # load csv
            df = load_data_from_csv(file_path)

            # extract the features
            df = _extract_features_from_signal(df, features_dict)

            # add class and subclass columns
            df = _add_class_and_subclass_column(df, folder_name, filename)

            # save in list
            df_list.append(df)

    # concat all dataframes
    all_data_df = pd.concat(df_list, ignore_index=True)

    # save to csv file
    output_path = os.path.join(output_path, output_filename)
    all_data_df.to_csv(output_path)

    # inform user
    print(f"Data saved to {output_path}")


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _extract_features_from_signal(df: pd.DataFrame, features_dict: Dict[Any, Any]) -> pd.DataFrame:
    # drop time column
    df.drop(columns=['sec'], inplace=True)


    # extract the features
    features_df = tsfel.time_series_features_extractor(features_dict, df, fs=100, window_size=240, overlap=0.5)

    return features_df


def _add_class_and_subclass_column(df: pd.DataFrame, folder_name: str, filename: str) -> pd.DataFrame:
    # atribute class and subclass numbers according to the activity and tasks
    if CABINETS in folder_name:
        class_number = 1
        subclass_number = _check_subclass(filename)

    elif SITTING in folder_name:
        class_number = 2
        subclass_number = _check_subclass(filename)

    elif STANDING in folder_name:
        class_number = 3
        subclass_number = _check_subclass(filename)

    elif WALKING in folder_name:
        class_number = 4
        subclass_number = _check_subclass(filename)

    else:
        raise ValueError(
            f"The activity: {folder_name} is not supported. Supported activities are: {SUPPORTED_ACTIVITIES}")

    # add class column
    df['class'] = class_number

    # add subclass column
    df['subclass'] = subclass_number

    return df


def _check_subclass(filename: str) -> float:
    # check subclass
    if COFFEE in filename:
        subclass_number = 1.1

    elif FOLDERS in filename:
        subclass_number = 1.2

    elif SIT in filename:
        subclass_number = 2.1

    elif GESTURES in filename:
        subclass_number = 3.1

    elif NO_GESTURES in filename:
        subclass_number = 3.2

    elif SLOW in filename:
        subclass_number = 4.1

    elif MEDIUM in filename:
        subclass_number = 4.2

    elif FAST in filename:
        subclass_number = 4.3

    else:
        raise ValueError(f"Subclass not supported. Supported subclasses are: ")

    return subclass_number
