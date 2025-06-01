import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from utils.config import EEG_SETTINGS, PLOTS_PATH, ANALYSIS_LOG_PATH, DATASETS, channel_positions, set_plot_style
from eeg_analyzer.dataset import Dataset

set_plot_style()

# Fix: use only the filename for the log file, not the full path
log_path = os.path.join(ANALYSIS_LOG_PATH, f"{os.path.basename(__file__)}.log")

def log(message, make_gap=False):
    """Log messages to file and print to stdout."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(__file__)
    if make_gap:
        message = "\n" + message
    log_entry = f"[{now}] [{filename}] {message}"
    print(log_entry)
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Ensure file exists (touch)
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            pass
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def prepare_datasets():
    """Load and return all datasets as Dataset objects."""
    braboszcz_config = DATASETS["braboszcz2017"]
    braboszcz = Dataset(braboszcz_config)
    jin_config = DATASETS["jin2019"]
    jin = Dataset(jin_config)
    touryan_config = DATASETS["touryan2022"]
    touryan = Dataset(touryan_config)
    return [braboszcz, jin, touryan]

def get_long_band_power_df(dataset, freq_band):
    """Get long-format band power dataframe for a dataset and frequency band."""
    epochs = dataset.to_long_band_power_list(
        freq_band=freq_band, 
        use_rois=False
    )
    return pd.DataFrame(epochs)


# ------------------- MAIN ANALYSIS LOGIC -------------------

# Frequency bands to analyze
freq_bands = {
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 100)
}

log("Preparing datasets...")
datasets = prepare_datasets()

for band, freqs in freq_bands.items():
    log(f"|||||||||||||||||||||||   ANALYZING BAND: {band}   |||||||||||||||||||||||", make_gap=True)
    for dataset in datasets:
        log(f"|||||||||||||||||||||||   ANALYZING DATASET: {dataset.name}   |||||||||||||||||||||||", make_gap=True)

        log(f"Loading in subjects...")
        tic = datetime.now()
        dataset.load_subjects()
        toc = datetime.now()
        log(f"Done loading in subjects. Took {toc - tic} seconds")
        log(f"\n{dataset}")

        # --- Step 1: Get long-format dataframe ---
        df_full = get_long_band_power_df(dataset, freqs)

        # --- Step 2: Split into channel dataframes ---
        df_channels = get_channel_dataframes(df_full)

        # --- Step 3: Z-score filtering ---
        df_channels_z = apply_zscore_filtering(df_channels)

        # --- Step 4: IQR filtering ---
        df_channels_iqr = apply_iqr_filtering(df_channels)

        # --- Step 5: Plotting (no filtering) ---
        log(f"Plotting boxplots without filtering...")
        path_to_box = os.path.join(PLOTS_PATH, dataset.f_name, "state-boxplots_per_subject_session", "no_filtering")
        plot_boxplots_for_subject_sessions(log, dataset, path_to_box, df_channels, band)

        # --- Step 6: Plotting (z-score filtering) ---
        log(f"Plotting boxplots with z-score filtering...")
        path_to_box = os.path.join(PLOTS_PATH, dataset.f_name, "state-boxplots_per_subject_session", "z_score_filtering")
        plot_boxplots_for_subject_sessions(log, dataset, path_to_box, df_channels_z, band, variant="z-score")

        # --- Step 7: Plotting (IQR filtering) ---
        log(f"Plotting boxplots with iqr filtering...")
        path_to_box = os.path.join(PLOTS_PATH, dataset.f_name, "state-boxplots_per_subject_session", "iqr_filtering")
        plot_boxplots_for_subject_sessions(log, dataset, path_to_box, df_channels_iqr, band, variant="iqr")

        log(f"|||||||||||||||||||||||   FINISHED ANALYZING DATASET: {dataset.name}   |||||||||||||||||||||||", make_gap=True)
        # Delete Dataset object to free up memory
        del dataset

log("Finished analyzing all datasets.")