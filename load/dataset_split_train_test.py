# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import load


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def train_test_split(file_path, train_size, test_size):
    """
    Splits the dataset into training and testing sets based on the given sizes.

    Args:
    - dataframes_list: DataFrame containing all the data.
    - train_size: Percentage of the data to be used for training (between 0 and 1).
    - test_size: Percentage of the data to be used for testing (between 0 and 1).

    Returns:
    - train_set: DataFrame containing the train set.
    - test_set: DataFrame containing the test set.
    """
    # separate the dataframes by subclass of movement
    dataframes_list = _split_by_subclass(file_path)

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


# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #
# path

def _split_by_subclass(file_path, subclass_column='subclass'):
    """
    Splits the DataFrame into sub-DataFrames based on the unique values
    in the subclass column and stores them in a list.

    Args:
    - df: The input DataFrame.
    - subclass_column: The column name containing the subclass information.

    Returns:
    - A list of sub-DataFrames, each corresponding to a unique subclass.
    """
    # load dataset
    df = load.load_data_from_csv(file_path)

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
