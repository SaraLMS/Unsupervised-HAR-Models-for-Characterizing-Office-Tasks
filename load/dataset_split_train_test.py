# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd

from load import load_data_from_csv


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def train_test_split(path: str, train_size, test_size):
    """
    Splits the dataset into training and testing sets based on the given sizes.
    The train set corresponds to the first train_size percentage of the dataset and the test_set the last
    test_size percentage of the dataset.

    Args:
    - dataframes_list: DataFrame containing all the data.
    - train_size: Percentage of the data to be used for training (between 0 and 1).
    - test_size: Percentage of the data to be used for testing (between 0 and 1).

    Returns:
    - train_set: DataFrame containing the train set.
    - test_set: DataFrame containing the test set.
    """

    df = load_data_from_csv(path)

    # separate the dataframes by subclass of movement
    dataframes_list = _split_by_subclass(df)

    # lists for holding the train and test sets for each subclass
    train_set_list = []
    test_set_list = []

    # iterate through the dataframes
    for dataframe in dataframes_list:
        # Determine the number of samples for training and testing
        train_end = int(len(dataframe) * train_size)
        test_start = int(len(dataframe) * (1 - test_size))

        # Split the dataframe into train and test
        train_df = dataframe.iloc[:train_end]
        test_df = dataframe.iloc[test_start:]

        # Append to the respective lists
        train_set_list.append(train_df)
        test_set_list.append(test_df)

    # concat list of dataframes to one
    train_set = pd.concat(train_set_list)
    test_set = pd.concat(test_set_list)

    return train_set, test_set


def unbalance_dataset(df, subclass_column='subclass', target_proportions=None):
    """
    Unbalances the dataset by ensuring that the final proportions of the dataset
    are 90% sitting, 5% walking, and 5% standing.

    Args:
    - subclass_dfs: List of dataframes, each corresponding to a different subclass.
    - subclass_column: The column name identifying the subclass in each DataFrame.
    - target_proportions: Dictionary specifying the target proportions.

    Returns:
    - A new DataFrame that is unbalanced according to the specified proportions.
    """
    subclass_dfs = _split_by_subclass(df)

    # Default proportions if not specified
    if target_proportions is None:
        target_proportions = {
            'sit': 0.90,
            'walk_medium': 0.05,
            'standing_still1': 0.05
        }

    # Extract and sum the rows for each subclass
    subclass_rows = {key: None for key in target_proportions.keys()}  # initialize dictionary
    for df in subclass_dfs:
        subclass_name = df[subclass_column].iloc[0]  # Assume each df has a 'subclass' column
        for key in subclass_rows:
            if subclass_name == key:
                subclass_rows[key] = df  # Store DataFrame under the correct category

    # Check if all necessary categories are available
    for key, value in subclass_rows.items():
        if value is None:
            raise ValueError(f"No data found for required subclass: {key}")

    # Total 'sit' rows determine the size of other categories
    total_sit_rows = len(subclass_rows['sit'])
    total_final_rows = total_sit_rows / target_proportions['sit']  # Calculate total rows based on sit proportion

    # Calculate needed rows for other subclasses
    final_dataframes = [subclass_rows['sit']]  # Start with all 'sit' rows
    for key, prop in target_proportions.items():
        if key != 'sit':  # Already handled 'sit'
            needed_rows = int(total_final_rows * prop)
            if len(subclass_rows[key]) < needed_rows:
                print(f"Warning: Not enough data for {key}. Taking all available rows.")
                final_dataframes.append(subclass_rows[key])
            else:
                final_dataframes.append(subclass_rows[key].head(needed_rows))

    # Concatenate all selected rows into a single DataFrame
    result_df = pd.concat(final_dataframes, ignore_index=True)

    print(result_df['subclass'].value_counts())

    return result_df


def load_basic_activities_only(path):

    dict = {}

    df = load_data_from_csv(path)

    # separate the dataframes by subclass of movement
    dataframes_list = _split_by_subclass(df)

    for dataframe in dataframes_list:

        if dataframe['subclass'].iloc[0] == "standing_still1":

            dict['standing_still1'] = dataframe

        elif dataframe['subclass'].iloc[0] == "standing_still2":

            dict['standing_still2'] = dataframe

        elif dataframe['subclass'].iloc[0] == "sit":

            dict['sitting'] = dataframe

        elif dataframe['subclass'].iloc[0] == "walk_medium":

            dict['walking'] = dataframe

        else:
            raise ValueError("Class not supported")

    # concat the standing still
    stand_still_list = [dict['standing_still1'], dict['standing_still2']]
    standing_df = pd.concat(stand_still_list)

    # remove the stand still 1 and 2 and add the merged dataframe
    dict.pop('standing_still1', None)
    dict.pop('standing_still2', None)
    dict['standing'] = standing_df

    walking_df = dict['walking']
    sitting_df = dict['sitting']

    print("Check results:")
    print(f"Sitting dataframe: len {len(sitting_df)}; class {sitting_df['class'].iloc[0]}")
    print(f"Standing dataframe: {len(standing_df)}; class {standing_df['class'].iloc[0]}")
    print(f"Walking dataframe: {len(walking_df)}; class {walking_df['class'].iloc[0]}")

    return dict



# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
# path


def _split_by_subclass(df, subclass_column='subclass'):
    """
    Splits the DataFrame into sub-DataFrames based on the unique values
    in the subclass column and stores them in a list.

    Args:
    -
    - subclass_column: The column name containing the subclass information.

    Returns:
    - A list of sub-DataFrames, each corresponding to a unique subclass.
    """

    # list to store dataframes of each subclass
    sub_dataframes = []

    # find the unique values in the subclass column
    subclasses = df[subclass_column].unique()

    for subclass in subclasses:
        # get a dataframe with only the values of each subclass
        sub_df = df[df[subclass_column] == subclass]

        # store in list
        sub_dataframes.append(sub_df)

    return sub_dataframes
