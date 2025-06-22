import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS

sns.set_style("ticks") 

def plot_distribution_comparison(data_before: np.ndarray, data_after: np.ndarray, 
                               title: str, save_path: str = None) -> None:
    """Plot distribution comparison before and after transformation/filtering."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Before transformation - histogram
    axes[0, 0].hist(data_before.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Before - Histogram')
    axes[0, 0].set_xlabel('Power [µV²]')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After transformation - histogram
    axes[0, 1].hist(data_after.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('After - Histogram')
    axes[0, 1].set_xlabel('Power [ln(µV²)]')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Before transformation - Q-Q plot
    stats.probplot(data_before.flatten(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Before - Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # After transformation - Q-Q plot
    stats.probplot(data_after.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('After - Q-Q Plot (Normal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    
    plt.close(fig)


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

    # Plotting for each subject, channel. Before and after log transfomation
    print("Plotting histograms for each subject and channel after log transformation...")
    output_dir_per_subject = os.path.join(analyzer.derivatives_path, "log-transformation", "per_subject_by_channel")
    os.makedirs(output_dir_per_subject, exist_ok=True)
    for dataset in analyzer:
        for subject in dataset:
            ch_names = subject.get_channel_names()
            ch_indices = [ch_names.index(ch) for ch in channels_to_plot if ch in ch_names]
            for recording in subject:
                log_band_power = recording.get_log_band_power()
                band_power = recording.get_band_power()
                outlier_mask = recording.get_outlier_mask()
                
                # Use .all(axis=1) to keep only rows with all valid channels:
                log_band_power_filtered = log_band_power[outlier_mask.all(axis=1), :]
                band_power_filtered = band_power[outlier_mask.all(axis=1), :]

                for ch, ch_idx in zip(channels_to_plot, ch_indices):
                    print(f"Plotting distribution before and after log transformation for channel {ch} in subject {subject.id} and session {recording.session_id} of dataset {dataset.name}")

                    log_data = log_band_power_filtered[:, ch_idx]
                    raw_data = band_power_filtered[:, ch_idx]

                    title = f"Data Distribution (Alpha Band Power) | Dataset: {dataset.name} | Subject: {subject.id} | Session: {recording.session_id} | Channel: {ch}"
                    plot_distribution_comparison(
                        data_before=raw_data,
                        data_after=log_data,
                        title=title,
                        save_path=os.path.join(output_dir_per_subject, f"{dataset.name}_subj{subject.id}_sess{recording.session_id}_ch{ch}.svg")
                    )
