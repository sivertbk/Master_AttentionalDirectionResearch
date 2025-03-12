"""
Configuration file for the EEG analysis project.
Defines global parameters, file paths, and default values.
"""

import os

# Project root directory (adjust if necessary)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# File paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
EPOCHS_PATH = os.path.join(DATA_PATH, "epochs")
PSD_DATA_PATH = os.path.join(DATA_PATH, "psd_data")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs")
PLOTS_PATH = os.path.join(PROJECT_ROOT, "plots")

# Logging
PREPROCESSING_LOG = os.path.join(LOGS_PATH, "preprocessing_logs", "preprocessing.log")
ANALYSIS_LOG = os.path.join(LOGS_PATH, "analysis_logs", "analysis.log")

# EEG Preprocessing Settings
LOW_CUTOFF_HZ = 1.0  # Low-frequency cutoff for high-pass filter
HIGH_CUTOFF_HZ = 40.0  # High-frequency cutoff for low-pass filter
ICA_COMPONENTS_TO_REMOVE = []  # List of components to be manually removed
EPOCH_LENGTH_SEC = 5.0  # Length of epochs in seconds

# PSD Settings
PSD_FREQUENCY_RES = 0.5  # Frequency resolution in Hz
PSD_NORMALIZATION = "z-score"  # Normalization method (options: "min-max", "z-score", None)

# Outlier Detection
Z_SCORE_THRESHOLD = 3.0  # Threshold for outlier detection based on z-score

# Visualization
FIGURE_DPI = 300  # Resolution for saved figures

if __name__ == "__main__":
    print("Configuration file loaded. Project root:", PROJECT_ROOT)

