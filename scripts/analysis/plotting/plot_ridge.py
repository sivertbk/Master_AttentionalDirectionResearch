import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde

from eeg_analyzer.eeg_analyzer import EEGAnalyzer
from utils.config import DATASETS, set_plot_style
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
                                "log_alpha_power": filtered_alpha_power[i, ch_idx]
                            })

    return pd.DataFrame(data_rows)

def plot_ridge(data: pd.DataFrame, ridge_slices, x_col='log_alpha_power', dist_cols=None, title="", output_path=None, show=False, return_fig=True, add_global_dist=False):
    """
    Creates a ridge plot from a DataFrame, inspired by seaborn's ridge plot examples.

    Args:
        data (pd.DataFrame): The input data.
        ridge_slices (str or list): The column(s) to define the ridges.
        x_col (str, optional): The column for the x-axis values. Defaults to 'log_alpha_power'.
        dist_cols (str or list, optional): Column(s) to separate distributions within a ridge. Defaults to None.
        title (str, optional): The plot title. Defaults to "".
        output_path (str, optional): Path to save the figure. If None, not saved. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
        return_fig (bool, optional): Whether to return the figure object. Defaults to True.
        add_global_dist (bool, optional): Whether to add a global distribution line for each ridge. Defaults to False.
    """
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

    # Initialize the FacetGrid object
    num_dists = len(data[dist_group_col].unique()) if dist_group_col else 1
    pal = sns.color_palette("viridis", num_dists)

    g = sns.FacetGrid(data, row=ridge_group_col, hue=dist_group_col, aspect=15, height=.5, palette=pal)

    # Draw the densities
    g.map(sns.kdeplot, x_col, bw_adjust=.5, clip_on=False, fill=True, alpha=0.5, linewidth=1.5)
    g.map(sns.kdeplot, x_col, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # Draw the global distribution if requested
    if add_global_dist:
        g.map(sns.kdeplot, x_col, clip_on=False, color="black", lw=1, bw_adjust=.5, linestyle="--")

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color='black', ha="left", va="center", transform=ax.transAxes)

    g.map(label, x_col)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.5)

    # Remove axes details
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

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
    ANALYZER_NAME = "eeg_analyzer_test"

    analyzer = EEGAnalyzer.load_analyzer(ANALYZER_NAME)
    if analyzer is None:
        print(f"Analyzer {ANALYZER_NAME} not found. Creating a new one.")
        analyzer = EEGAnalyzer(DATASETS, ANALYZER_NAME)
        analyzer.save_analyzer()

    data = gather_data(analyzer)

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
                   dist_cols=['task', 'state'],
                   add_global_dist=True,
                   x_col='log_alpha_power',
                   title=f"Log Alpha Power Distribution for {dataset.name}", 
                   output_path=output_path, show=False
                   )