import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import EEGANALYZER_SETTINGS, set_plot_style
set_plot_style()  # Set the plotting style for matplotlib

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
                    alpha_power = recording.get_log_band_power(task, state)
                    outlier_mask = recording.get_outlier_mask(task, state)

                    # Determine which epochs to keep (non-outliers)
                    if outlier_mask is not None:
                        # Indices of epochs that are NOT outliers
                        epoch_indices = np.where(~outlier_mask)[0]
                        # Filter the alpha power data to only include non-outlier epochs
                        filtered_alpha_power = alpha_power[epoch_indices, :]
                    else:
                        # If no outlier mask is present, keep all epochs
                        epoch_indices = np.arange(alpha_power.shape[0])
                        filtered_alpha_power = alpha_power

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
                                "log_band_power": filtered_alpha_power[i, ch_idx]
                            })

    return pd.DataFrame(data_rows)

def plot_ridge(data: pd.DataFrame, ridge_slices, x_col='log_band_power', dist_cols=None, title="", output_path=None, show=False, return_fig=True, add_global_dist=False, overlap=0.5):
    """
    Creates a ridge plot from a DataFrame, inspired by seaborn's ridge plot examples.

    Args:
        data (pd.DataFrame): The input data.
        ridge_slices (str or list): The column(s) to define the ridges.
        x_col (str, optional): The column for the x-axis values. Defaults to 'log_band_power'.
        dist_cols (str or list, optional): Column(s) to separate distributions within a ridge. Defaults to None.
        title (str, optional): The plot title. Defaults to "".
        output_path (str, optional): Path to save the figure. If None, not saved. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
        return_fig (bool, optional): Whether to return the figure object. Defaults to True.
        add_global_dist (bool, optional): Whether to add a global distribution line for each ridge. Defaults to False.
        overlap (float, optional): Controls the overlap of the ridges. A value of 0 means no overlap, a positive value creates overlap. Defaults to 0.5.
    """
    # Make a copy to avoid SettingWithCopyWarning
    data = data.copy()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    if isinstance(ridge_slices, str):
        ridge_slices = [ridge_slices]

    # Create a combined column for unique ridge identification
    if len(ridge_slices) > 1:
        data['ridge_group'] = data[ridge_slices].apply(lambda x: '-'.join(x.astype(str)), axis=1)
        ridge_group_col = 'ridge_group'
    else:
        ridge_group_col = ridge_slices[0]

    # Create a combined column for distributions if multiple are given
    dist_group_col = None
    if dist_cols:
        if isinstance(dist_cols, str):
            dist_cols = [dist_cols]
        if len(dist_cols) > 1:
            data['dist_group'] = data[dist_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
            dist_group_col = 'dist_group'
        else:
            dist_group_col = dist_cols[0]

    # Use a more distinct color palette
    num_dists = len(data[dist_group_col].unique()) if dist_group_col else 1
    pal = sns.color_palette("Set2", num_dists)

    g = sns.FacetGrid(data, row=ridge_group_col, hue=dist_group_col, aspect=15, height=.5, palette=pal)

    # Draw the densities
    g.map(sns.kdeplot, x_col, bw_adjust=.6, clip_on=False, fill=True, alpha=0.6, linewidth=1.5)
    # Draw a white line over them to create separation
    #g.map(sns.kdeplot, x_col, clip_on=False, color="w", lw=2, bw_adjust=.6)
    # Draw a line at the bottom of each plot
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Set the labels for each ridge using the ridge slice names
    for i, ax in enumerate(g.axes.flat):
        ax.text(0.02, 0.2, g.row_names[i], fontweight="bold", color='black',
                ha="left", va="center", transform=ax.transAxes)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-overlap)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add the global distribution if requested
    if add_global_dist:
        for i, ax in enumerate(g.axes.flat):
            ridge_label = g.row_names[i]
            ridge_data = data[data[ridge_group_col] == ridge_label]
            sns.kdeplot(data=ridge_data, x=x_col, ax=ax, color="#565656FF", linestyle=":", lw=1.5, bw_adjust=.6, clip_on=False)

    if title:
        g.fig.suptitle(title, y=1.02, fontsize=16)

    if dist_group_col:
        g.add_legend(title='-'.join(dist_cols))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='svg', bbox_inches='tight')

    if show:
        plt.show()

    fig = g.fig
    if not return_fig:
        plt.close(fig)
        return None
    
    return fig


if __name__ == "__main__":
    eeganalyzer_kwargs = EEGANALYZER_SETTINGS.copy()
    ANALYZER_NAME = EEGANALYZER_SETTINGS["analyzer_name"]

    # Trying to load the EEGAnalyzer
    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(**eeganalyzer_kwargs)
        analyzer.save_analyzer()

    analyzer.create_dataframe(exclude_bad_recordings=False)
    data = analyzer.df

    if data is None:
        print("No valid data available for plotting.")
        exit()

    # Making ridge plots for a subset of subjects in each dataset over one specified channel
    # the ridgeback is made up by the subset of subjects (i.e. ridge_slice is 'subject_id') and the channel
    # This is useful for comparing subjects within a dataset
    channel_to_plot = 'Cz'  # Specify the channel to plot
    n_subjects_per_plot = 32  # Max number of subjects to plot per dataset

    for dataset in analyzer:
        # Filter data for the current dataset and the specified channel
        dataset_channel_data = data[(data['dataset'] == dataset.name) & (data['channel'] == channel_to_plot)]

        if dataset_channel_data.empty:
            print(f"No data for channel {channel_to_plot} in dataset {dataset.name}. Skipping subject ridge plot.")
            continue

        # Get a subset of subjects to plot
        unique_subjects = dataset_channel_data['subject_id'].unique()
        subjects_to_plot = unique_subjects[:n_subjects_per_plot]

        if len(subjects_to_plot) == 0:
            continue

        # Filter the data for the selected subjects
        plot_data = dataset_channel_data[dataset_channel_data['subject_id'].isin(subjects_to_plot)]

        # Define the output path for the ridge plot
        output_path = os.path.join(analyzer.derivatives_path, "ridge_plots", f"{dataset.name}_channel-{channel_to_plot}_subjects_ridge_plot.svg")

        print(f"Plotting ridge plot for {len(subjects_to_plot)} subjects from dataset {dataset.name} on channel {channel_to_plot}.")

        # Create the ridge plot
        plot_ridge(plot_data,
                   ridge_slices=['subject_id'],
                   dist_cols='state',
                   add_global_dist=True,
                   x_col='log_band_power',
                   title=f"Log Alpha Power on Ch. {channel_to_plot} for {dataset.name}",
                   output_path=output_path, show=False
                   )

    # Plotting ridge plot over all datasets
    if data.empty:
        print("No data available for plotting. Exiting.")
    else:
        # Define the output path for the combined ridge plot
        output_path = os.path.join(analyzer.derivatives_path, "ridge_plots", "combined_ridge_plot.svg")

        # remove every other channel to avoid cluttering the plot
        channels_to_remove = data['channel'].unique()[1::2]
        data = data[~data['channel'].isin(channels_to_remove)]
        
        # Create the combined ridge plot
        plot_ridge(data, 
                   ridge_slices=['channel'], 
                   dist_cols=None,
                   add_global_dist=False,
                   x_col='log_band_power',
                   title="Log Alpha Power Distribution Across Datasets", 
                   output_path=output_path, show=False
                   )

    # Plotting the ridge plot for each dataset
    for dataset in analyzer:
        dataset_data = data[data['dataset'] == dataset.name]
        if dataset_data.empty:
            print(f"No data available for dataset {dataset.name}. Skipping ridge plot.")
            continue
        
        # Define the output path for the ridge plot
        output_path = os.path.join(analyzer.derivatives_path, "ridge_plots", f"{dataset.name}_ridge_plot.svg")
        
        # Create the ridge plot
        plot_ridge(dataset_data, 
                   ridge_slices=['channel'], 
                   dist_cols='state',
                   add_global_dist=True,
                   x_col='log_band_power',
                   title=f"Log Alpha Power Distribution for {dataset.name}", 
                   output_path=output_path, show=False
                   )
        
        subject_data = None  # Reset subject_data for each dataset
        # Plotting for a single pseudorandom subject in each dataset
        for subject in dataset.subjects:
            if subject_data is not None:
                continue
            subject_data = dataset_data[dataset_data['subject_id'] == subject]
            if subject_data.empty:
                print(f"No data available for subject {subject} in dataset {dataset.name}. Skipping ridge plot.")
                continue

            # Define the output path for the ridge plot
            output_path = os.path.join(analyzer.derivatives_path, "ridge_plots", f"{dataset.name}_sub-{subject}_ridge_plot.svg")

            # Create the ridge plot
            plot_ridge(subject_data,
                    ridge_slices=['channel'],
                    dist_cols='state',
                    add_global_dist=True,
                    x_col='log_band_power',
                    title=f"Log Alpha Power Distribution for {dataset.name} - Subject-{subject}",
                    output_path=output_path, show=False
                    )