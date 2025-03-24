"""
Global configuration file for EEG Analysis Project.
Defines project paths, default settings, and global parameters.
"""

import os
from dataclasses import dataclass

# Import dataset-specific configurations
from utils.dataset_configs import jin2019_config, braboszcz2017_config

# =============================================================================
#                                 PATH SETTINGS
# =============================================================================

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
EPOCHS_PATH = os.path.join(DATA_PATH, "epochs")
PSD_DATA_PATH = os.path.join(DATA_PATH, "psd_data")

REPORTS_PATH = os.path.join(ROOT_PATH, "reports")
LOGS_PATH = os.path.join(REPORTS_PATH, "logs")
PREPROCESSING_LOG_PATH = os.path.join(LOGS_PATH, "preprocessing_logs")
ANALYSIS_LOG_PATH = os.path.join(LOGS_PATH, "analysis_logs")
PLOTS_PATH = os.path.join(REPORTS_PATH, "plots")

SCRIPTS_PATH = os.path.join(ROOT_PATH, "scripts")
UTILS_PATH = os.path.join(ROOT_PATH, "utils")

PREPROCESSING_LOG = os.path.join(PREPROCESSING_LOG_PATH, "preprocessing.log")
ANALYSIS_LOG = os.path.join(ANALYSIS_LOG_PATH, "analysis.log")


# =============================================================================
#                             GLOBAL EEG SETTINGS
# =============================================================================

EEG_SETTINGS = {
    "LOW_CUTOFF_HZ": 1.0,
    "HIGH_CUTOFF_HZ": 40.0,
    "ICA_COMPONENTS_TO_REMOVE": [],
    "EPOCH_LENGTH_SEC": 5.0,
    "PSD_FREQUENCY_RES": 0.5,
    "PSD_NORMALIZATION": "z-score",
    "Z_SCORE_THRESHOLD": 3.0,
}

VISUALIZATION_SETTINGS = {
    "FIGURE_DPI": 300
}

# =============================================================================
#                        DATASET SPECIFIC CONFIGURATIONS
# =============================================================================

from utils.dataset_configs import jin2019_config, braboszcz2017_config

DATASETS = {
    "Jin2019": jin2019_config,
    "Braboszcz2017": braboszcz2017_config
}

if __name__ == "__main__":
    print("Global configuration loaded successfully.")

