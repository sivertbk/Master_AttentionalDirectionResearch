import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS

def gather_data(analyzer: EEGAnalyzer = None):
    """
    Gathers the alpha power data in micro volts squared per Hz and decibels
    for each subject and channel in the EEG data.
    """
    if analyzer is None:
        raise ValueError("Analyzer must be provided.")

    data_rows = []
    for dataset in analyzer:
        for subject in dataset:
            for recording in subject:
                #if recording.exclude:
                #    continue
                
                ch_names = recording.get_channel_names()
                condition_list = recording.list_conditions()

                for task, state in condition_list:
                    # Gather log alpha power data and outlier mask
                    log_alpha_power = recording.get_log_band_power(task, state)
                    alpha_power = recording.get_band_power(task, state)
                    outlier_mask = recording.get_outlier_mask(task, state)

                    # Determine which epochs to keep (non-outliers)
                    if outlier_mask is not None:
                        # Indices of epochs that are NOT outliers
                        epoch_indices = np.where(~outlier_mask)[0]
                        # Filter the alpha power data to only include non-outlier epochs
                        filtered_log_alpha_power = log_alpha_power[epoch_indices, :]
                        filtered_alpha_power = alpha_power[epoch_indices, :]
                    else:
                        # If no outlier mask is present, keep all epochs
                        epoch_indices = np.arange(alpha_power.shape[0])
                        filtered_alpha_power = alpha_power
                        filtered_log_alpha_power = log_alpha_power

                    # Create a row for each epoch and channel
                    for i, epoch_idx in enumerate(epoch_indices):
                        for ch_idx, ch_name in enumerate(ch_names):
                            data_rows.append({
                                "dataset": dataset.name,
                                "subject": subject.id,
                                "group": subject.group,
                                "session": recording.session_id,
                                "epoch_idx": epoch_idx,
                                "task": task,
                                "state": state,
                                "channel": ch_name,
                                "alpha_power": filtered_alpha_power[i, ch_idx],
                                "log_alpha_power": filtered_log_alpha_power[i, ch_idx]
                            })

    return pd.DataFrame(data_rows)


def dot_histogram(data, mask=None, bins=10, colors=('red', 'blue'), dot_size=20, title=None):
    data = np.asarray(data)
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    
    # Compute histogram bin edges
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Assign each point to a bin
    bin_indices = np.digitize(data, bin_edges[:-1], right=False)
    
    # For each bin, collect the points and stack them vertically
    for b in range(1, bins+1):
        in_bin = bin_indices == b
        bin_values = data[in_bin]
        bin_mask = mask[in_bin]
        
        # Sort values in bin for consistent vertical stacking
        x = bin_values
        y = np.arange(1, len(x)+1)
        c = np.where(bin_mask, colors[1], colors[0])  # Use mask to assign color
        
        if len(x) > 0:
            plt.scatter(x, y, c=c, s=dot_size, edgecolor='k', linewidth=0.3, alpha=0.7)

    plt.xlabel("Value")
    plt.ylabel("Stacked count per bin")
    plt.title(title if title else "Dot Histogram with Outlier Masking")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def sns_dot_histogram(data, mask=None, bins=30, palette=('gray', 'crimson'),
                      dot_size=40, jitter=0.05, ax=None, figsize=(10, 5), 
                      alpha=0.8, title=None):
    """
    Plot a dot-based histogram with Seaborn styling and masked dot coloring.
    
    Parameters:
        data (array-like): 1D array of continuous values.
        mask (array-like): Boolean array same length as data. True for masked (e.g. outlier).
        bins (int): Number of histogram bins.
        palette (tuple): Two colors (normal, masked).
        dot_size (int): Size of each dot.
        jitter (float): Horizontal jitter to avoid overplotting.
        ax (matplotlib Axes): Optional. Use existing axis.
        figsize (tuple): If no axis given, set figure size.
        alpha (float): Dot transparency.
        title (str): Title of the plot.
    """
    sns.set(style="whitegrid")
    data = np.asarray(data)
    
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        mask = np.asarray(mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute histogram bin edges
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_indices = np.digitize(data, bin_edges[:-1], right=False)

    for b in range(1, bins + 1):
        in_bin = bin_indices == b
        bin_values = data[in_bin]
        bin_mask = mask[in_bin]
        
        x = bin_values
        y = np.arange(1, len(x) + 1)
        colors = np.where(bin_mask, palette[1], palette[0])
        
        if len(x) > 0:
            jittered_x = x + np.random.uniform(-jitter, jitter, size=len(x))
            ax.scatter(jittered_x, y, c=colors, s=dot_size,
                       alpha=alpha, edgecolor='k', linewidth=0.2)

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Count (stacked)", fontsize=12)
    ax.set_title(title if title else "Dot Histogram with Masked Values", fontsize=14)
    sns.despine(ax=ax)
    plt.tight_layout()
    return ax

def stacked_dot_histogram(data, mask=None, bins=30, palette=('gray', 'crimson'),
                          dot_size=40, ax=None, figsize=(10, 5), alpha=0.8, title=None):
    """
    Plot a stacked dot histogram where dots are grouped into bins, stacked vertically,
    and colored based on a boolean mask. Dots with mask=False are plotted last (on top).
    
    Parameters:
        data (array-like): 1D array of values.
        mask (array-like): Boolean array, True=normal, False=highlight/outlier.
        bins (int): Number of histogram bins.
        palette (tuple): Colors for (True, False) in the mask.
        dot_size (int): Size of each dot.
        ax (matplotlib Axes): Optional external axis to plot into.
        figsize (tuple): Size of the figure if ax is not provided.
        alpha (float): Dot transparency.
        title (str): Title of the plot.
    """
    sns.set(style="whitegrid")
    data = np.asarray(data)
    if mask is None:
        mask = np.ones_like(data, dtype=bool)
    else:
        mask = np.asarray(mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute histogram bin edges
    print("Creating histogram bins...")
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(f"Bin edges: {bin_edges}")
    print(f"Bin centers: {bin_centers}")
    print(f"Counts per bin: {counts}")

    # Assign data points to bins
    print("Assigning data points to bins...")
    bin_indices = np.digitize(data, bin_edges, right=False)
    
    for i in range(1, len(bin_edges)):
        print(f"Processing bin {i} with edges {bin_edges[i-1]} to {bin_edges[i]}")
        in_bin = bin_indices == i
        bin_data = data[in_bin]
        bin_mask = mask[in_bin]
        
        # Sort so that mask=True (normal) is first, False (outlier) is last
        order = np.argsort(~bin_mask)  # Invert mask to get desired order
        sorted_mask = bin_mask[order]
        
        x_pos = bin_centers[i - 1]
        for y_idx, is_normal in enumerate(sorted_mask, start=1):
            color = palette[0] if is_normal else palette[1]
            ax.scatter(x_pos, y_idx, color=color, s=dot_size,
                       alpha=alpha, edgecolor='k', linewidth=0.2)

    ax.set_xlabel("Alpha Power [ln(µV²)]", fontsize=12)
    ax.set_ylabel("Count (stacked)", fontsize=12)
    ax.set_title(title if title else "Stacked Dot Histogram with Outlier Overlay", fontsize=14)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    sns.despine(ax=ax)
    plt.tight_layout()
    return ax


if __name__ == "__main__":
    ANALYZER_NAME = "eeg_analyzer"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)
        analyzer.save_analyzer()

    # Prepare for plotting
    # Selecting a few channels to plot data from
    channels_to_plot = ['Cz', 'Pz', 'Oz', 'Fz', 'C3', 'C4', 'P3', 'P4'] # Using P3/P4 instead of T7/T8 as they are more common

    # Plotting for each subject, with subplots for each channel
    print("\nPlotting histograms for each subject and channel...")
    output_dir_per_subject = os.path.join(analyzer.derivatives_path, "histograms_outliers", "per_subject_by_channel")
    os.makedirs(output_dir_per_subject, exist_ok=True)
    for dataset in analyzer:
        for subject in dataset:
            ch_names = subject.get_channel_names()
            ch_indices = [ch_names.index(ch) for ch in channels_to_plot if ch in ch_names]
            data_list = []
            mask_list = []
            for recording in subject:
                log_band_power = recording.get_log_band_power()
                outlier_mask = recording.get_outlier_mask()
                data_list.append(log_band_power[:, ch_indices])
                mask_list.append(outlier_mask[:, ch_indices])
            data_combined = np.concatenate(data_list, axis=0)
            mask_combined = np.concatenate(mask_list, axis=0)

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(f"Dataset: {dataset.name}, Subject: {subject.id}", fontsize=16)
            for i, ch_name in enumerate(channels_to_plot):
                if ch_name not in ch_names:
                    continue
                ax = axes[i // 4, i % 4]
                stacked_dot_histogram(data_combined[:, i],
                                      mask=mask_combined[:, i],
                                      bins=30, ax=ax, title=ch_name)
            plt.savefig(os.path.join(output_dir_per_subject,
                                     f"{dataset.name}_subj{subject.id}.svg"),
                        format='svg', bbox_inches='tight')
            plt.close(fig)

    print(f"Saved plots to {output_dir_per_subject}")



    # Plotting across subjects for each channel
    print("\nPlotting histograms across subjects for each channel...")
    output_dir_across = os.path.join(analyzer.derivatives_path, "histograms_outliers", "across_subjects")
    os.makedirs(output_dir_across, exist_ok=True)
    for dataset in analyzer:
        data_to_plot = []
        mask_to_plot = []
        for subject in dataset:
            ch_names = subject.get_channel_names()
            ch_indices = [ch_names.index(ch) for ch in channels_to_plot if ch in ch_names]
            for recording in subject:
                log_band_power = recording.get_log_band_power()
                outlier_mask = recording.get_outlier_mask()
                
                # Collect data and mask for plotting
                data_to_plot.append(log_band_power[:, ch_indices])
                mask_to_plot.append(outlier_mask[:, ch_indices])
        data_to_plot = np.concatenate(data_to_plot, axis=0)
        mask_to_plot = np.concatenate(mask_to_plot, axis=0)

        # Create one figure with 2x4 subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Across-subject Histograms - {dataset.name}")

        for ch_idx, ch_name in enumerate(channels_to_plot):
            if ch_name not in ch_names:
                continue
            ax = axes[ch_idx // 4, ch_idx % 4]
            stacked_dot_histogram(data_to_plot[:, ch_idx],
                                  mask=mask_to_plot[:, ch_idx],
                                  bins=80, ax=ax,
                                  title=f"{ch_name} across subjects")

        plt.savefig(os.path.join(output_dir_across, f"{dataset.name}_all_channels.svg"),
                    format='svg', bbox_inches='tight')
        plt.close(fig)

    print(f"Saved plots to {output_dir_across}")




