"""
Global configuration file for EEG Analysis Project.
Defines project paths, default settings, and global parameters.
"""

import os
from dataclasses import dataclass
from utils.helpers import calculate_freq_resolution

# =============================================================================
#                                 PATH SETTINGS
# =============================================================================

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASETS_PATH          = os.path.join(DATA_PATH, "datasets")
RAW_DATA_PATH          = os.path.join(DATA_PATH, "raw")
EPOCHS_PATH            = os.path.join(DATA_PATH, "epochs")
PSD_DATA_PATH          = os.path.join(DATA_PATH, "psd_data")

REPORTS_PATH           = os.path.join(ROOT_PATH, "reports")
LOGS_PATH              = os.path.join(REPORTS_PATH, "logs")
PREPROCESSING_LOG_PATH = os.path.join(LOGS_PATH, "preprocessing_logs")
ANALYSIS_LOG_PATH      = os.path.join(LOGS_PATH, "analysis_logs")
PLOTS_PATH             = os.path.join(REPORTS_PATH, "plots")

SCRIPTS_PATH           = os.path.join(ROOT_PATH, "scripts")
UTILS_PATH             = os.path.join(ROOT_PATH, "utils")

PREPROCESSING_LOG      = os.path.join(PREPROCESSING_LOG_PATH, "preprocessing.log")
ANALYSIS_LOG           = os.path.join(ANALYSIS_LOG_PATH, "analysis.log")


# =============================================================================
#                             GLOBAL EEG SETTINGS
# =============================================================================

EEG_SETTINGS = {
    "LOW_CUTOFF_HZ": 1.0,
    "HIGH_CUTOFF_HZ": 40.0,
    "N_ICA_COMPONENTS": 20,
    "EPOCH_LENGTH_SEC": 5.0,
    "EPOCH_START_SEC": -5.0,
    "PSD_NORMALIZATION": "z-score",
    "Z_SCORE_THRESHOLD": 3.0,
    "SAMPLING_RATE": 128.0,
    "REJECT_THRESHOLD": 150e-6,
    "PSD_WINDOW": "hann",
    "PSD_OVERLAP_RATIO": 0.5,
    "PSD_FMIN": 4.0,
    "PSD_FMAX": 40.0,
    "PSD_REMOVE_DC": True,
}

VISUALIZATION_SETTINGS = {
    "FIGURE_DPI": 300
}



# Calculate frequency resolution and derive PSD parameters
EEG_SETTINGS["PSD_FREQ_RESOLUTION"] = calculate_freq_resolution(EEG_SETTINGS["EPOCH_LENGTH_SEC"])
EEG_SETTINGS["PSD_N_FFT"] = int(EEG_SETTINGS["SAMPLING_RATE"] / EEG_SETTINGS["PSD_FREQ_RESOLUTION"])
EEG_SETTINGS["PSD_N_OVERLAP"] = int(EEG_SETTINGS["PSD_N_FFT"] * EEG_SETTINGS["PSD_OVERLAP_RATIO"])


# =============================================================================
#                        DATASET SPECIFIC CONFIGURATIONS
# =============================================================================

from utils.dataset_configs.jin2019_config import jin2019_config
from utils.dataset_configs.braboszcz2017_config import braboszcz2017_config

# Set dataset paths dynamically:
for dataset in [jin2019_config, braboszcz2017_config]:
    dataset.path_raw    = os.path.join(RAW_DATA_PATH, dataset.f_name)
    dataset.path_epochs = os.path.join(EPOCHS_PATH, dataset.f_name)
    dataset.path_psd    = os.path.join(PSD_DATA_PATH, dataset.f_name)

DATASETS = {
    "jin2019": jin2019_config,
    "braboszcz2017": braboszcz2017_config
}

if __name__ == "__main__":
    print("Global configuration loaded successfully.")

