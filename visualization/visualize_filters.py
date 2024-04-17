# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import glob
import os

from load.load_sync_data import load_data_from_csv
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def visualize_filtering_results(filtered_signals_dict, sync_data_path):
    """
    Plots and compares raw and filtered sensor data side by side for each sensor.
    This function finds the raw data file that includes the folder name in its filename, loads the data,
    and then plots both the raw and filtered data for comparison.

    Parameters:
        filtered_signals_dict (dict): A dictionary where keys are folder names, and values are DataFrames of filtered data.
        sync_data_path (str): The base directory path containing folders of raw CSV signal data files.

    Notes:
        Expects filenames in each folder to include the folder name as part of a larger filename structure.
    """
    for folder_name, filtered_df in filtered_signals_dict.items():
        # Search for the CSV file in the folder that contains the folder name in its filename
        folder_path = os.path.join(sync_data_path, folder_name)
        file_pattern = os.path.join(folder_path, f"*{folder_name}*.csv")
        matching_files = glob.glob(file_pattern)

        if not matching_files:
            print(f"No matching files found in {folder_path} for pattern {file_pattern}")
            continue

        # Assume the first match is the correct file (update logic here if necessary)
        file_path = matching_files[0]
        raw_df = load_data_from_csv(file_path)

        # Setup plot
        num_sensors = len(raw_df.columns)
        fig, axs = plt.subplots(num_sensors, 1, figsize=(10, num_sensors * 3), sharex=True)
        fig.suptitle(f'Filtering Comparison for {folder_name}')

        # Ensure axs is iterable
        if num_sensors == 1:
            axs = [axs]

        # Plot each sensor
        for idx, sensor in enumerate(raw_df.columns):
            axs[idx].plot(raw_df.index, raw_df[sensor], label='Raw', color='blue', alpha=0.75)
            axs[idx].plot(filtered_df.index, filtered_df[sensor], label='Filtered', color='green', alpha=0.75)
            axs[idx].set_title(sensor)
            axs[idx].legend()

        plt.xlabel('Time (indices)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()