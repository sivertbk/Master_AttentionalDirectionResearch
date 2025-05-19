import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from utils.config import EEG_SETTINGS, PLOTS_PATH, ANALYSIS_LOG_PATH, DATASETS, channel_positions, ROIs, set_plot_style
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
    """Sort channels by y-coordinate (posterior [lowest y] to anterior [highest y])."""
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

def mark_bad_datapoints(df, method="z-score", threshold=3.0):
    """
    Mark bad datapoints at the channel level using the specified filtering method.

    Parameters:
        df (DataFrame): Long-format dataframe with channel-level data.
        method (str): Filtering method ("z-score" or "iqr").
        threshold (float): Threshold for z-score filtering (ignored for IQR).

    Returns:
        DataFrame: Updated dataframe with a new column 'is_bad' (True for bad datapoints).
    """
    df['is_bad'] = False

    if method == "z-score":
        grouped = df.groupby(['subject_session', 'state', 'channel'])
        z_scores = grouped['band_power'].transform(lambda x: (x - x.mean()) / x.std())
        df['is_bad'] = z_scores.abs() > threshold

    elif method == "iqr":
        def mark_iqr(group):
            q1 = group['band_power'].quantile(0.25)
            q3 = group['band_power'].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            group['is_bad'] = ~group['band_power'].between(lower, upper)
            return group

        df = df.groupby(['subject_session', 'state', 'channel'], group_keys=False).apply(mark_iqr)
    return df

def aggregate_to_rois(df, roi_mapping, handle_bad="exclude_epoch"):
    """
    Aggregate channel-level data to ROI-level data.

    Parameters:
        df (DataFrame): Long-format dataframe with channel-level data.
        roi_mapping (dict): Nested mapping of ROI names to sub-ROIs, each containing lists of channels.
        handle_bad (str): How to handle bad datapoints ("exclude_epoch" or "replace_bad").

    Returns:
        DataFrame: ROI-level aggregated dataframe.
    """
    roi_data = []

    # Flatten the nested ROI mapping to get a single mapping of ROI to channels
    flattened_roi_mapping = {
        roi: channels
        for roi, sub_rois in roi_mapping.items()
        for sub_roi, channels in sub_rois.items()
    }

    for (subject_session, state, epoch_idx), group in df.groupby(['subject_session', 'state', 'epoch_idx']):

        for roi, channels in flattened_roi_mapping.items():
            roi_group = group[group['channel'].isin(channels)]

            if roi_group.empty:
                continue

            if handle_bad == "exclude_epoch":
                if roi_group['is_bad'].any():
                    continue  # Exclude this epoch for the ROI
                mean_power = roi_group['band_power'].mean()

            elif handle_bad == "replace_bad":
                good_data = roi_group[~roi_group['is_bad']]
                if good_data.empty:
                    continue  # Skip if all channels are bad
                mean_power = good_data['band_power'].mean()
                roi_group.loc[roi_group['is_bad'], 'band_power'] = mean_power

            roi_data.append({
                "subject_session": subject_session,
                "state": state,
                "epoch_idx": epoch_idx,
                "roi": roi,
                "band_power": mean_power
            })

    return pd.DataFrame(roi_data)

def plot_roi_boxplot(log, dataset, path_to_box, df_roi, band, variant=None):
    """
    Plot and save a boxplot for all ROIs for a subject-session.

    Parameters:
        log (function): Logging function.
        dataset (Dataset): Dataset object.
        path_to_box (str): Path to save the boxplot.
        df_roi (DataFrame): ROI-level dataframe.
        band (str): Frequency band name.
        variant (str): Filtering variant (e.g., "z-score", "iqr").
    """
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    state_palette = [cmap(0.0), cmap(1.0)]
    state_labels = {0: "Focused", 1: "Mind-wandering"}
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=df_roi,
        x="roi",
        y="band_power",
        hue="state",
        palette=state_palette,
        dodge=True
    )
    ax.set_xlabel("Region of Interest (ROI)")
    ax.set_ylabel(f"{band} Power [µV²]")
    variant_title = f"- {variant}-filtered" if variant is not None else ""
    ax.set_title(f"{band} Power by State {variant_title}\nAggregated by ROI")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [state_labels[int(l)] for l in labels], title="State")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Prepare save path
    fname = f"{dataset.f_name}_roi_{band.lower()}_{variant.lower()}-filtered_boxplot.png" if variant else f"{dataset.f_name}_roi_{band.lower()}_boxplot.png"
    save_dir = os.path.join(path_to_box, band.lower())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path)
    plt.close()
    log(f"Saved ROI-level boxplot: {save_path}")

# ------------------- MAIN ANALYSIS LOGIC -------------------

# Define ROI mapping 
roi_mapping = ROIs

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

        # --- Step 2: Mark bad datapoints ---
        df_full = mark_bad_datapoints(df_full, method="z-score", threshold=3.0)

        # --- Step 3: Aggregate to ROIs ---
        df_roi = aggregate_to_rois(df_full, roi_mapping, handle_bad="exclude_epoch")
        #log the head of the dataframe
        log(f"ROI-level dataframe head:\n{df_roi.head()}")

        # --- Step 4: Plotting ROI-level boxplots ---
        log(f"Plotting ROI-level boxplots...")
        path_to_box = os.path.join(PLOTS_PATH, dataset.f_name, "state-boxplots_per_roi")
        plot_roi_boxplot(log, dataset, path_to_box, df_roi, band, variant="z-score")

        log(f"|||||||||||||||||||||||   FINISHED ANALYZING DATASET: {dataset.name}   |||||||||||||||||||||||", make_gap=True)
        # Delete Dataset object to free up memory
        del dataset

log("Finished analyzing all datasets.")