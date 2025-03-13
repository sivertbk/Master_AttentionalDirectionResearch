"""
Configuration file for the EEG analysis project.
Defines global parameters, file paths, and default values.
"""

import os

# Project root directory (one level up from the current file's location)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# File paths
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
EPOCHS_PATH = os.path.join(DATA_PATH, "epochs")
PSD_DATA_PATH = os.path.join(DATA_PATH, "psd_data")

# Reports and logs
REPORTS_PATH = os.path.join(ROOT_PATH, "reports")
LOGS_PATH = os.path.join(REPORTS_PATH, "logs")
PREPROCESSING_LOG_PATH = os.path.join(LOGS_PATH, "preprocessing_logs")
ANALYSIS_LOG_PATH = os.path.join(LOGS_PATH, "analysis_logs")
PLOTS_PATH = os.path.join(REPORTS_PATH, "plots")

# Scripts and utilities
SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")
UTILS_PATH = os.path.join(ROOT_PATH, "utils")

# Logging
PREPROCESSING_LOG = os.path.join(PREPROCESSING_LOG_PATH, "preprocessing.log")
ANALYSIS_LOG = os.path.join(ANALYSIS_LOG_PATH, "analysis.log")



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
    print("Configuration file loaded. Project root:", ROOT_PATH)
