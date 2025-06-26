import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS

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




if __name__ == "__main__":
    eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
    ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
        analyzer.save_analyzer()

    print("Gathering data...")
    data = gather_data(analyzer)
    print("Data gathered.")

    # Prepare for plotting
    # Selecting a few channels to plot data from
    channels_to_plot = ['Cz', 'Pz', 'Oz', 'Fz', 'C3', 'C4', 'P3', 'P4'] # Using P3/P4 instead of T7/T8 as they are more common

    # Plotting across subjects for each channel, split by condition
    print("\nPlotting histograms across subjects for each channel...")
    output_dir_across = os.path.join(analyzer.derivatives_path, "histograms", "across_subjects_by_condition")
    os.makedirs(output_dir_across, exist_ok=True)

    for channel in channels_to_plot:
        channel_data = data[data['channel'] == channel]
        if channel_data.empty:
            print(f"No data available for channel {channel}. Skipping.")
            continue

        plt.figure(figsize=(12, 6))
        sns.histplot(data=channel_data, x='log_alpha_power', bins=10, kde=True, element="step")
        plt.title(f"Log Alpha Power Distribution - Channel {channel} (Across Subjects)")
        plt.xlabel("Log Alpha Power ln(µV²)")
        plt.ylabel("Count")
        plt.grid(True)
        
        # Saving svg to derivative folder
        plt.savefig(os.path.join(output_dir_across, f"log_alpha_power_dist_{channel}_by_condition.svg"), format='svg', bbox_inches='tight')
        plt.close()
    print(f"Saved plots to {output_dir_across}")


    # Plotting for each subject, with subplots for each channel, split by condition
    print("\nPlotting histograms for each subject and channel...")
    output_dir_per_subject = os.path.join(analyzer.derivatives_path, "histograms", "per_subject_by_channel")
    os.makedirs(output_dir_per_subject, exist_ok=True)

    for subject in data['subject'].unique():
        subject_data = data[data['subject'] == subject]
        if subject_data.empty:
            print(f"No data available for subject {subject}. Skipping.")
            continue

        # Create a grid of subplots
        n_channels = len(channels_to_plot)
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharex=True, sharey=True)
        axes = axes.flatten() # Flatten to 1D array for easy iteration

        for i, channel in enumerate(channels_to_plot):
            ax = axes[i]
            # Filter data for the specific subject and channel
            plot_data = subject_data[subject_data['channel'] == channel]

            if not plot_data.empty:
                sns.histplot(data=plot_data, x='log_alpha_power', bins=15, kde=True, element="step", ax=ax, legend= i == 0)
                ax.set_title(f"Channel {channel}")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f"No data for {channel}", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].set_visible(False)

        # Common labels and title
        fig.suptitle(f"Log Alpha Power Distribution - Subject {subject}", fontsize=20)
        fig.supxlabel("Log Alpha Power ln(µV²)", fontsize=14)
        fig.supylabel("Count", fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle

        # Saving svg to derivative folder
        plt.savefig(os.path.join(output_dir_per_subject, f"log_alpha_power_dist_subject_{subject}.svg"), format='svg', bbox_inches='tight')
        plt.close(fig)
    print(f"Saved plots to {output_dir_per_subject}")

