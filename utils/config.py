"""
Global configuration file for EEG Analysis Project.
Defines project paths, default settings, and global parameters.
"""

import os
import matplotlib.pyplot as plt

from utils.helpers import calculate_freq_resolution

# =============================================================================
#                                 PATH SETTINGS
# =============================================================================

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(ROOT_PATH, "data")

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
    "LOW_CUTOFF_HZ": 0.5, 
    "HIGH_CUTOFF_HZ": 40.0,
    "N_ICA_COMPONENTS": None,
    "EPOCH_LENGTH_SEC": 5.0,
    "EPOCH_START_SEC": -5.0,
    "SYNTHETIC_LENGTH": 2,
    "PSD_NORMALIZATION": "z-score",
    "Z_SCORE_THRESHOLD": 3.0,
    "SAMPLING_RATE": 128.0,
    "REJECT_THRESHOLD": 150e-6,
    "PSD_UNIT_CONVERT": 1e12, # Convert V²/Hz to uV²/Hz
    "PSD_OUTPUT": "power",
    "PSD_WINDOW": "hann",
    "PSD_AVERAGE_METHOD": "mean",
    "PSD_N_FFT": 256,
    "PSD_N_PER_SEG": 128,
    "PSD_N_OVERLAP": 64,
    "PSD_FMIN": 4.0,
    "PSD_FMAX": 40.0,
    "PSD_REMOVE_DC": True,
    "MONTAGE": "biosemi64",
    "AR_MAX_TRAINING": 5000000 # Maximum number of epochs to train autoreject. Set to inf to use all epochs.
}

EEG_SETTINGS["PSD_FREQ_RESOLUTION"] = calculate_freq_resolution(EEG_SETTINGS["SAMPLING_RATE"], EEG_SETTINGS["PSD_N_FFT"])



def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (6, 4),  # default figure size
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Font
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],  # fallback chain
        "font.size": 11,

        # Axes
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid
        "axes.grid": False,
        "grid.color": "#dddddd",
        "grid.linestyle": "--",

        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,

        # Color cycle
        "axes.prop_cycle": plt.cycler(color=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]),

        # PDF/PS output
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })




# EEG 10-20 channel positions (x, y, z)
channel_positions = {
    "Fp1": (-27, 83, -3),
    "AF7": (-51, 71, -3),
    "AF3": (-36, 76, 24),
    "F1": (-25, 62, 56),
    "F3": (-48, 59, 44),
    "F5": (-64, 55, 23),
    "F7": (-71, 51, -3),
    "FT7": (-83, 27, -3),
    "FC5": (-78, 30, 27),
    "FC3": (-59, 31, 56),
    "FC1": (-33, 33, 74),
    "C1": (-34, 0, 81),
    "C3": (-63, 0, 61),
    "C5": (-82, 0, 31),
    "T7": (-87, 0, -3),
    "TP7": (-83, -27, -3),
    "CP5": (-78, -30, 27),
    "CP3": (-59, -31, 56),
    "CP1": (-33, -33, 74),
    "P1": (-25, -62, 56),
    "P3": (-48, -59, 44),
    "P5": (-64, -55, 23),
    "P7": (-71, -51, -3),
    "P9": (-64, -47, -37),
    "PO7": (-51, -71, -3),
    "PO3": (-36, -76, 24),
    "O1": (-27, -83, -3),
    "Iz": (0, -79, -37),
    "Oz": (0, -87, -3),
    "POz": (0, -82, 31),
    "Pz": (0, -63, 61),
    "CPz": (0, -34, 81),
    "Fpz": (0, 87, -3),
    "Fp2": (27, 83, -3),
    "AF8": (51, 71, -3),
    "AF4": (36, 76, 24),
    "AFz": (0, 82, 31),
    "Fz": (0, 63, 61),
    "F2": (25, 62, 56),
    "F4": (48, 59, 44),
    "F6": (64, 55, 23),
    "F8": (71, 51, -3),
    "FT8": (83, 27, -3),
    "FC6": (78, 30, 27),
    "FC4": (59, 31, 56),
    "FC2": (33, 33, 74),
    "FCz": (0, 34, 81),
    "Cz": (0, 0, 88),
    "C2": (34, 0, 81),
    "C4": (63, 0, 61),
    "C6": (82, 0, 31),
    "T8": (87, 0, -3),
    "TP8": (83, -27, -3),
    "CP6": (78, -30, 27),
    "CP4": (59, -31, 56),
    "CP2": (33, -33, 74),
    "P2": (25, -62, 56),
    "P4": (48, -59, 44),
    "P6": (64, -55, 23),
    "P8": (71, -51, -3),
    "P10": (64, -47, -37),
    "PO8": (51, -71, -3),
    "PO4": (36, -76, 24),
    "O2": (27, -83, -3),
}

# EEG scalp regions using 10-20 layout
scalp_regions = {
    "frontal": [
        "Fp1", "Fp2", "Fpz",
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"
    ],
    "fronto-temporal": [
        "FT7", "FT8"
    ],
    "fronto-central": [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6"
    ],
    "central": [
        "C1", "C3", "Cz", "C2", "C4"
    ],
    "centro-parietal": [
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"
    ],
    "temporal": [
        "T7", "T8"
    ],
    "temporo-parietal": [
        "TP7", "TP8"
    ],
    "parietal": [
        "P1", "P3", "P5", "P7", "Pz", "P2", "P4", "P6", "P8"
    ],
    "posterior-parietal": [
        "P9", "P10"
    ],
    "parieto-occipital": [
        "PO7", "PO3", "POz", "PO4", "PO8"
    ],
    "occipital": [
        "O1", "Oz", "O2", "Iz"
    ]
}

# Scalp region coords
scalp_region_coords = {
    "frontal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["frontal"]},
    "fronto_temporal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["fronto-temporal"]},
    "central_coords" : {ch: channel_positions[ch] for ch in scalp_regions["central"]},
    "centro_parietal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["centro-parietal"]},
    "temporal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["temporal"]},
    "temporo_parietal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["temporo-parietal"]},
    "parietal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["parietal"]},
    "posterior_parietal_coords" : {ch: channel_positions[ch] for ch in scalp_regions["posterior-parietal"]},
    "parieto_occipital_coords" : {ch: channel_positions[ch] for ch in scalp_regions["parieto-occipital"]},
    "occipital_coords" : {ch: channel_positions[ch] for ch in scalp_regions["occipital"]}
}





# =============================================================================
#                        DATASET SPECIFIC CONFIGURATIONS
# =============================================================================

from utils.dataset_configs import DATASETS

# Set dataset paths dynamically based on the new per-dataset folder structure
for dataset in DATASETS.values():
    dataset_dir = os.path.join(DATA_PATH, dataset.f_name)

    dataset.path_root   = dataset_dir
    dataset.path_raw    = os.path.join(dataset_dir, "raw")
    dataset.path_epochs = os.path.join(dataset_dir, "epochs")
    dataset.path_psd    = os.path.join(dataset_dir, "psd_data")
    dataset.path_derivatives = os.path.join(dataset_dir, "derivatives")


if __name__ == "__main__":
    print("Global configuration loaded successfully.")
    for dataset in DATASETS.values():
        print(f"Dataset: {dataset.name}")
        print(f"  Root Path: {dataset.path_root}")
        print(f"  Raw Path: {dataset.path_raw}")
        print(f"  Epochs Path: {dataset.path_epochs}")
        print(f"  PSD Path: {dataset.path_psd}")
        print(f"  Derivatives Path: {dataset.path_derivatives}")

