# Unsupervised Human Activity Recognition Models for Characterizing Office Work Tasks

## Table of Contents
- [About](#about)
- [Synchronization](#synchronization)
- [Signal Pre-processing](#Signal_Preprocessing)
- [Feature Extraction](#Feature_Extraction)
- [Experiments](#Experiments)
  - [Experiment 1 - Feature Selection](#Experiment1_Feature_Selection)
  - [Experiment 2 - Cluster Stability](#Experiment2_Cluster_Stability)
  - [Experiment 3 - Cluster Imbalance](#Experiment3_Cluster_Imbalance)

## About 
The python code available in this project is part of the methods for the Master Thesis
"Unsupervised Human Activity Recognition Models for Characterizing Office Work Tasks".
In the following sections, the synchronization, pre-processing, and feature extraction pipelines utilized in this work will be explained.
For Human Activity Recognition (HAR) in office environments, three experiments were conducted: feature selection, cluster stability, and data imbalance.

In this thesis, data was acquired from 19 subjects performing 9 different tasks. The tasks were analysed hierarchically as follows:
![Diagram](./Figures/hier_HAR.png)

Five acquisition sessions were devised, grouping similar tasks within the same recording. At the start of each acquisition, subjects performed ten vertical short jumps to later synchronize the signals from the different devices. Short segments were performed in between tasks to later allow for segmentation.
The y-axis accelerometer signals of the five sessions are shown bellow:

![Diagram](./Figures/Acquisition_protocol_plot.png)

Check *main.py*, *run_experiment1.py*, *run_experiment2.py*, and *run_experiment3.py* for examples.

## Synchronization

![Diagram](./Figures/synchronization_before_after.png)

The "**synchronization**" function loads, synchronizes android sensors from the same device, resamples the signals, and synchronizes between two devices. To use the following code effectively, data should be acquired using the PrevOccupAI APP.

There are two stages to synchronize multiple android sensors from different devices:
1. **Synchronize android sensors within the same device**. Using the timestamps of the first sensor to start and the last to stop acquiring;
2. **Synchronize sensors between devices**. The following methods are supported:
   + *Cross-correlation*: synchronizes based on the maximum cross-correlation from the Accelerometer (ACC) signals from the two devices: y-axis for the smartphone and -x-axis for the smartwatch. For this method to work correctly, for each acquisition the subject must perform an initial segment of vertical short jumps, with the arms straight and parallel to the body.
   
   + *Timestamps*: synchronizes the signals based on the timestamps of the  first sample received from each device, contained in the logger file (opensignals). If this file does not exist or does not have the necessary timestamps, the timestamps in the filenames are used.


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

## Signal_Preprocessing

The *processing* package has three python files: task_segmentation, filters, and pre_processing. The *processor* or *process_all* functions in the *pre_processing* file apply the task segmentation method and the filters to the signals.
This function attributes filenames depending on the segment being segmented.

For task segmentation, the tasks performed within the same recording are segmented if the user performs ten-second stops between different walking activities and ten-second stops with a jump in the middle for standing activities.

To segment walking tasks, an onset-based task segmentation was implemented, while for standing and walking tasks, a peak-based approach was implemented. Some thresholds in the *segment_tasks* function in *task_segmentation.py* might need adjustments.

For filtering, the *apply_filters* in *pre_processing.py* applies a Butterworth lowpass filter to remove high frequency noise, a median filter for smoothing, and another lowpass filter to isolate the gravitational component which is then subtracted from the ACC signals. The cutoff frequencies and window lengths can be adjusted in the functions in *filters.py*.


The *processor* function has the following parameters:

+ sync_data_path (str): Path to the folder (i.e., sync_devices) containing the synchronized data main_folder/subfolder/sync_devices/sync_data.csv
+ output_base_path(str): Path to the base path were the raw segments and filtered segments should be saved
+ device_sensors_foldername (str): Name of the folder containing the loaded sensors and devices (i.e., acc_gyr_mag_phone_watch)
+ sub_folder (str): Name of the subfolder which contains the synchronized data main_folder/subfolder/sync_devices/sync_data.csv
+ raw_folder_name (str): (default = raw_tasks) Name of the folder where to store the raw signal segments
+ filtered_folder_name (str): (default = filtered_tasks) Name of the folder where to store the filtered signal segments
+ save_raw_tasks (bool): (default = True) Save the raw signal segments. False not to save
+ fs (int): sampling frequency
+ impulse_response_samples (int): Number of samples to be cut at the start of each segment to remove the impulse response of the filters

## Feature_Extraction

The TSFEL package (https://tsfel.readthedocs.io/en/latest/) was utilized for windowing and feature extraction. 
The *feature_extractor* function in *feature_engineering.feature_extraction.py* extracts features from sensor data files contained within the sub-folders of a main directory, as follows:
    main_dir/subfolders/sync_signals.csv.
1. Loads the signals to a pandas dataframe
2. Applies a sliding window on the columns of the dataframe (signals) and extracts the features chosen in the
    cfg_file.json. Check TSFEL documentation here: https://tsfel.readthedocs.io/en/latest/
3. Adds a class and subclass column based on the original file name
4. Balances the dataset to ensure the same amount of data from each class. Within each class, the subclass instances
    are also balanced to ensure, approximately, the same amount of data from each subclass. Each subclass should be
    equally sampled inside their respective class (the signals from each subclass should have the same duration) for
    this function to work correctly.
5. Saves the dataframe to a csv file where the columns are the feature names and the class and subclass, and the
    rows are the data points. The file name is generated automatically with addition to the prefix and suffix.

This function has the following parameters:

+ data_main_path (str): Path to the folder. Signals are contained in the sub folders inside the main path. For example:
        devices_folder_name/*folder*/subfolders/sync_signals.csv
+ output_path (str): Path to the folder where the csv file should be saved.
+ subclasses (List[str]): List containing the name of the subclasses to load and extract features. Supported subclasses:
            "sit": sitting
            "standing_still": Standing still
            "standing_gestures": Standing with gestures
            "coffee": Standing while doing coffee
            "folders": Standing while moving folders inside a cabinet
            "walk_slow": Walking slow speed
            "walk_medium": Walking medium speed
            "walk_fast": Walking fast speed
            "stairs": Going up and down the stairs
+ json_path (str): Path to the json file containing the features to be extracted using TSFEL
+ Prefix (str): String to be added at the start of the output filename
+ Suffix (str): String to be added at the end of the output filename
+ devices_folder_name (str): String to be used to form the output filename. Should be indicative of the sensors and devices used. If using "feature_extraction_all" this should be the name of the main folder as follows:
        *devices_folder_name*/folder/subfolders/sync_signals.csv
        *acc_gyr_mag_phone*/filtered_tasks/walking/walk_slow_signals_filename.csv

## Experiments

The models available for the following experiments are KMeans, Agglomerative Clustering (AGG), and Gaussian Mixture Model (GMM).
More clustering models can be added in *models.py*. Update the valid models in *constants.py*.

Three different models were implemented:

+ Subject-specific models (SSM): Cluster each subject individually with an optimized feature set

+ One-stage general model (1GM): Combine all subjects data for clustering

+ Two-stage general model (2GM): Cluster each subject individually with a common feature set

## Experiment1_Feature_Selection

Experiment 1 comprised of feature selection. To apply the one-stage feature selection method for the SSM and the 1GM
run the *run_experiment1.py* as follows:

    with open('run_experiment1.py') as f:
        code = f.read()
        exec(code)

In this file, the user chooses which model (SSM, 1GM) to run this experiment by setting the booleans to True or False.
For the 1GM, the *feature_selector* function works as follows:
This function returns the feature sets that produce the best clustering results for the train set
as well as the adjusted rand index of the respective feature sets. This feature selection method works as follows:
1. Normalize the features between 0 and 1, using MinMaxScaler from sklearn
2. Remove low variance and highly correlated features, given the variance_threshold and correlation_threshold
3. Shuffle the remaining features and add iteratively to a subset
4. If the Adjusted Rand Index(ARI) increases at least 1 % the feature is kept, if not, it is removed. This process
    produces a plot where the y-axis contains the metrics and the x-axis the feature added in each iteration, in order
    to analyze the most relevant features. This plot is saved if save_plots is set to True
5. Repeat step 1-4 n_iterations times to account for the randomness introduced by the shuffling.

The parameters are:
+ train_set (pd.DataFrame): Train set containing the features extracted (columns) and data instances (rows).
+ variance_threshold (float): Minimum variance value. Features with variance lower than this threshold will be removed.
+ correlation_threshold (float): Maximum correlation value. Removes features with a correlation higher or equal to this threshold
+ n_iterations (int): Number of times the feature selection method is repeated.
+ clustering_model (str): Unsupervised learning model used to select the best features. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering
        "gmm": Gaussian Mixture Model
+ output_path (str): Path to the main folder in which the plots should be saved
+ folder_name (str): Name of the folder in which to store the plots. Default: "phone_features_kmeans_plots".
+ save_plots (bool): If True, saves the plots of each iteration of the feature selection process. Don't save if False.

This function returns a list of feature sets, and correspondent list of ARI and NMI.

For the SSM, the *one_stage_feature_selection* function calls the feature_selector function which returns a list of feature sets (list[str]), and the
correspondent Adjusted Rand Index (List[float]) and Normalized Mutual Information (List[NMI]). Next, the feature
set with the highest ARI is selected. If more than 1, then the one with the highest NMI is chosen. If there is still
multiple feature sets with the same ARI and NMI, the feature set with the lowest number of features is chosen.
If in this case there are still multiple sets, then one is chosen randomly.

1. the subject id is extracted from the filename. For this, the filenames should have a capital P letter followed
    by three digits as follows: P001. This is then added in the first column
    The results are saved in a .txt file as follows:
2. the best feature set obtained is saved in the second column
3. the third and fourth columns of the txt file contain the ARI and NMI obtained, respectively.

For the 2GM, a two-stage feature selection method is implemented, which consists of applying the one-stage method, then,
from the subject-specific sets, the most common features are selected to form the final feature set. The used selects how many
features the final feature set should have. The *two_stage_feature_selection* function is used for this model.

## Experiment2_Cluster_Stability

In this experiment, clustering is performed on the train set and the test set instances are appointed to the
preformed clusters. If Agglomerative Clustering is chosen, clustering is performed on 20 % of the data.
Similar to experiment 1, this run experiment 2 as follows:

    with open('run_experiment3.py') as f:
        code = f.read()
        exec(code)

In *run_experiment3.py*, select the booleans to True depending on which model to run (SSM, 1GM, and 2GM).
For the SSM, *subject_specific_clustering* function does the following:
The feature sets used for this model are specific for each subject (or dataset) and are loaded from a txt file
(subjects_features_path) which has two columns: the first is the subject id, and the second is the best feature
set found for the respective subject in experiment 1, separated by ';' (i.e., P001; ['xAcc_Mean', 'zMag_Max'])
This function saves the adjusted rand index, normalized mutual information, and accuracy score (random forest)
results in an Excel sheet. This function has the following parameters:

+ main_path (str): Path to the main_folder containing subfolders which have the datasets. The directory scheme is the following
    main_folder/folder/subfolder/dataset.csv
    (i.e., datasets/features_basic_acc_gyr_mag_phone_watch/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)
+ subjects_features_path (str): Path to the txt file containing the features sets for each subject. These should match the same conditions of the
    dataset which will be used for this experiment (i.e., if clustering all activities, the path should be to the
    feature sets found for all activities). Otherwise, this the results obtained will not be correct.
+ clustering_model (str): Unsupervised learning model used for clustering. Supported models are:
        "kmeans" - KMeans clustering
        "agglomerative": Agglomerative clustering model
        "gmm": Gaussian Mixture Model
+ nr_clusters (int): Number of clusters to find
+ features_folder_name: str:  Path to the *folder* identifying which datasets to load. The directory scheme is the following
    *folder*/subfolder/dataset.csv
    (i.e., features_basic_acc_gyr_mag_phone_watch*/P001/features_basic_acc_gyr_mag_phone_watch_P001.csv)
+ results_path (str): Path to the directory where to save the Excel sheet with the clustering and random forest results.

For the 1GM and the 2GM, the *one_stage_general_model_each_subject* and *two_stage_general_model_clustering* functions
are used. These two function differ from the SSM since the user must input the feature set (List[str]).

## Experiment3_Cluster_Imbalance

To run experiment 3, the user must the *run_experiment3.py*:

    with open('run_experiment3.py') as f:
        code = f.read()
        exec(code)

The *imbalanced_clustering* function applies a sliding window approach on a dataframe in order to extract multiple consecutive chunks of
the standing still and walking medium data, as shown in the figure bellow. These chunks are then added to the sitting data to form the final
dataset. The clustering result is obtained  by doing the mean ARI over all datasets. With this function, 
the user can select the final sitting proportion, as well as the number of chunks from the standing
and walking instances to have.

![Diagram](./Figures/imbalanced_datasets.png)




