# Unsupervised Human Activity Recognition Models for Characterizing Office Work Tasks

## Table of Contents
- [About](#about)
- [Synchronization](#synchronization)
- [Signal Pre-processing](#processing)
- [Feature Extraction](#feature_extraction)
- [Experiments](#experiments)
  - [Experiment 1 - Feature Selection](#experiment1)
  - [Experiment 2 - Cluster Stability](#experiment2)
  - [Experiment 3 - Cluster Imbalance](#experiment3)

## About 
This project focuses on Human Activity Recognition (HAR) using unsupervised 
learning models.

## Synchronization

![Diagram](./figures/synchronization_before_after.png)


There are two stages to synchronize multiple android sensors from different devices:
1. Synchronize android sensors within the same device. Using the timestamps of the first sensor to start and the last to stop acquiring;
2. Synchronize sensors between devices. The following methods are supported:
   + Cross-correlation: synchronizes based on the maximum cross-correlation from the Accelerometer (ACC) signals from the two devices: y-axis for the smartphone and -x-axis for the smartwatch. For this method to work correctly, for each acquisition the subject must perform an initial segment of vertical short jumps, with the arms straight and parallel to the body.
   + Timestamps: synchronizes the signals based on the timestamps of the  first sample received from each device, contained in the logger file (opensignals). If this file does not exist or does not have the necessary timestamps, the timestamps in the filenames are used.


This is done using the *synchronization* function. This function allows the user to choose
which sensor data to load and synchronize. If the user chooses sensors from only one
device, only these are synchronized. This function has the following parameters:

+ raw_data_in_path (str): path to the main folder containing subfolders with raw sensor data (i.e., ../main_folder/subfolders/sensor_data.txt)
+ sync_android_out_path (str): path to the location where the synchronized android sensor data, within each device, will be stored.
+ selected_sensors (Dict[str, List[str]): dictionary containing the devices and sensors chosen to be loaded and synchronized (i.e., selected_sensors = {'phone': ['acc', 'gyr', 'mag'], 'watch': ['acc', 'gyr', 'mag']}).
+ output_path (str): path to the location where the file containing the synchronized data, from the multiple devices, should be saved.
+ sync_type (str): method for synchronizing data between devices (crosscorr or timestamps).
+ evaluation_filename (str): name of the file which will contain the synchronization evaluation report.
+ save_intermediate_files (bool): keep the csv files generated after synchronizing android
        sensors. False to delete. If there's only signals from one device, these files are not deleted.







