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

There are two stages to synchronize multiple android sensors from different devices:
1. Synchronize android sensors within the same device. Using the timestamps of the first sensor to start and the last to stop acquiring;
2. Synchronize sensors between devices. The following methods are supported:
   + Cross-correlation: synchronizes based on the maximum cross-correlation from the Accelerometer (ACC) signals from the two devices: y-axis for the smartphone and -x-axis for the smartwatch;
   + Timestamps: synchronizes the signals based on the timestamps of the  first sample received from each device, contained in the logger file (opensignals). If this file does not exist or does not have the necessary timestamps, the timestamps in the filenames are used.


This is done using the *synchronization* function. This function allows the user to choose
which sensor data to load and synchronize. If the user chooses sensors from only one
device, only these are synchronized.


