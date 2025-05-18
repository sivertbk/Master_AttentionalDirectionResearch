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

def plot_channel_boxplot(log, dataset, path_to_box, subject_session, df_ss, sorted_channels, band, variant=None):
    """Plot and save a boxplot for all channels for a subject-session."""
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    state_palette = [cmap(0.0), cmap(1.0)]
    state_labels = {0: "Focused", 1: "Mind-wandering"}
    plt.figure(figsize=(max(5, len(sorted_channels) * 0.5), 8))
    ax = sns.boxplot(
            data=df_ss,
            x="channel",
            y="band_power",
            hue="state",
            palette=state_palette,
            order=sorted_channels,
            hue_order=[0, 1],
            dodge=True
        )
    ax.set_xlabel("")  # No axis label for x
    ax.set_ylabel(f"{band} Power [µV²]")
    variant_title = f"- {variant}-filtered" if variant is not None else ""
    ax.set_title(f"{band} Power by State {variant_title}\nSubject-Session: {subject_session}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [state_labels[int(l)] for l in labels], title="State")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    # Add text under x-axis, centered
    ax.annotate(
            "Posterior <--> Anterior",
            xy=(0.5, -0.18),
            xycoords='axes fraction',
            ha='center',
            va='center',
            fontsize=12
        )
    plt.tight_layout()

    # Prepare save path
    if variant is not None:
        fname = f"{dataset.f_name}_{subject_session}_allchannels_{band.lower()}_{variant.lower()}-filtered_boxplot.png"
    else:
        fname = f"{dataset.f_name}_{subject_session}_allchannels_{band.lower()}_boxplot.png"
    save_dir = os.path.join(path_to_box, band.lower())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path)
    plt.close()
    log(f"Saved all-channels boxplot: {save_path}")

def iqr_filter_within_state(df):
    """Remove outliers using IQR for each (subject_session, state) group."""
    def filter_group(group):
        q1 = group['band_power'].quantile(0.25)
        q3 = group['band_power'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group['band_power'] >= lower) & (group['band_power'] <= upper)]
    return df.groupby(['subject_session', 'state'], group_keys=False).apply(filter_group)

def z_score_within_state(df):
    """Add a z-score column within each (subject_session, state) group."""
    grouped = df.groupby(['subject_session', 'state'])
    df['z_score'] = grouped['band_power'].transform(lambda x: (x - x.mean()) / x.std())
    return df

def get_channel_dataframes(df_full):
    """Split a long-format dataframe into a list of dataframes, one per channel."""
    return [df_full[df_full['channel'] == channel].reset_index(drop=True) for channel in df_full['channel'].unique()]

def apply_zscore_filtering(df_channels):
    """Apply z-score filtering to a list of channel dataframes."""
    df_channels_z = []
    for df_channel in df_channels:
        df_channel_z = z_score_within_state(df_channel)
        df_channels_z.append(df_channel_z[df_channel_z['z_score'] <= 3])
    return df_channels_z

def apply_iqr_filtering(df_channels):
    """Apply IQR filtering to a list of channel dataframes."""
    return [iqr_filter_within_state(df_channel) for df_channel in df_channels]

def get_sorted_channels(df_ss):
    """Sort channels by y-coordinate (dorsal [lowest y] to ventral [highest y])."""
    unique_channels = df_ss['channel'].unique()
    sorted_channels = sorted(
        unique_channels,
        key=lambda ch: channel_positions.get(ch, (0, 0, 0))[1]
    )
    return sorted_channels

def plot_boxplots_for_subject_sessions(
    log, dataset, path_to_box, df_channels, band, variant=None
):
    """Plot boxplots for all subject-sessions for a given filtering variant."""
    os.makedirs(path_to_box, exist_ok=True)
    all_subject_sessions = pd.concat(df_channels).subject_session.unique()
    for subject_session in all_subject_sessions:
        # Collect data for all channels for this subject_session
        df_ss = pd.concat([
            df_channel[df_channel['subject_session'] == subject_session]
            for df_channel in df_channels
        ])
        if df_ss.empty:
            continue
        sorted_channels = get_sorted_channels(df_ss)
        df_ss['channel'] = pd.Categorical(df_ss['channel'], categories=sorted_channels, ordered=True)
        plot_channel_boxplot(log, dataset, path_to_box, subject_session, df_ss, sorted_channels, band, variant=variant)

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